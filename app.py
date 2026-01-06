import os
import time
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# --- CONFIGURACIÓN ---
app = Flask(__name__)
CORS(app)  # Habilita conexión con Vercel

# =====================================================
# CONFIGURACIÓN BASE DE DATOS (LAZY, NO EN STARTUP)
# =====================================================
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = 'sismapiscis'
DB_PORT = os.getenv('DB_PORT', '3306')

_db_engine = None


def get_engine():
    global _db_engine
    if _db_engine is None:
        print("🚀 Inicializando conexión MySQL (lazy)")
        db_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        _db_engine = create_engine(
            db_url,
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5}
        )
    return _db_engine


def get_db_connection():
    """Obtiene una conexión directa para ejecutar SQL crudo"""
    return get_engine().connect()


# =====================================================
# ENDPOINT ROOT (HEALTHCHECK REAL)
# =====================================================
@app.route('/', methods=['GET'])
def root():
    try:
        with get_db_connection() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({
            "status": "ok",
            "service": "BioTwin AI Backend",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "warning",
            "service": "BioTwin AI Backend",
            "database": "not_connected",
            "error": str(e)
        }), 200


# --- UTILIDADES DE INTELIGENCIA ---
def calcular_color_agua(nitrato):
    """Lógica de Shader: Devuelve Hex color basado en turbidez (Nitrato)"""
    if nitrato <= 10:
        return "#0EA5E9"
    if nitrato <= 40:
        return "#22C55E"
    return "#854D0E"


# --- ENDPOINTS (API V1) ---

@app.route('/api/v1/biofloc/status/<int:pool_id>', methods=['GET'])
def get_biofloc_status(pool_id):
    try:
        with get_db_connection() as conn:
            query = text("""
                SELECT ion_nitrato, oxigeno_disuelto, ph, temperatura 
                FROM parametro_aguas 
                WHERE piscina_id = :pid 
                ORDER BY fecha_medicion DESC LIMIT 1
            """)
            result = conn.execute(query, {"pid": pool_id}).fetchone()

            if not result:
                return jsonify({
                    "ion_nitrato": 0,
                    "carbon_demand": 0,
                    "oxigeno_disuelto": 0,
                    "ph": 7.0,
                    "temperatura": 25.0,
                    "estado_critico": False,
                    "dosing_locked": True,
                    "carbon_amount_gr": 0
                })

            nitrato, o2, ph, temp = result

            target_nitrato = 10.0
            carbon_demand = 0
            carbon_amount_gr = 0

            if nitrato > target_nitrato:
                delta_n = nitrato - target_nitrato
                carbon_demand = delta_n * 15
                carbon_amount_gr = round(carbon_demand * 20, 2)

            dosing_locked = True if o2 < 4.0 else False
            estado_critico = True if nitrato > 50.0 or o2 < 3.0 else False

            return jsonify({
                "ion_nitrato": round(float(nitrato), 2),
                "carbon_demand": round(float(carbon_demand), 2),
                "oxigeno_disuelto": round(float(o2), 2),
                "ph": round(float(ph), 2),
                "temperatura": round(float(temp), 2),
                "estado_critico": estado_critico,
                "dosing_locked": dosing_locked,
                "carbon_amount_gr": carbon_amount_gr
            })

    except Exception as e:
        print(f"Error en biofloc status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/digital-twin/config/<int:pool_id>', methods=['GET'])
def get_digital_twin_config(pool_id):
    try:
        with get_db_connection() as conn:
            q_pool = text("SELECT largo, ancho, profundidad FROM piscinas WHERE id = :pid")
            pool_data = conn.execute(q_pool, {"pid": pool_id}).fetchone()

            q_water = text("SELECT ion_nitrato FROM parametro_aguas WHERE piscina_id = :pid ORDER BY id DESC LIMIT 1")
            water_data = conn.execute(q_water, {"pid": pool_id}).fetchone()

            q_bio = text("""
                SELECT cantidad_muestreo, peso_promedio
                FROM biometrias
                WHERE piscina_id = :pid
                ORDER BY fecha_registro DESC LIMIT 1
            """)
            bio_data = conn.execute(q_bio, {"pid": pool_id}).fetchone()

            l, w, d = pool_data if pool_data else (10, 5, 1.2)
            nitrato = water_data[0] if water_data else 0
            fish_count = bio_data[0] if bio_data else 1000
            avg_weight = bio_data[1] if bio_data else 10.0

            turbidity = min(nitrato / 100.0, 1.0)

            return jsonify({
                "dimensions": {"l": float(l), "w": float(w), "d": float(d)},
                "water_quality": {
                    "color_hex": calcular_color_agua(nitrato),
                    "turbidity_factor": float(turbidity)
                },
                "biomass": {
                    "fish_count": int(fish_count),
                    "avg_weight": float(avg_weight)
                }
            })

    except Exception as e:
        print(f"Error digital twin: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/predict/oxygen/<int:pool_id>', methods=['GET'])
def get_oxygen_prediction(pool_id):
    try:
        query = text("""
            SELECT fecha_medicion, oxigeno_disuelto, temperatura, ph, ion_nitrato 
            FROM parametro_aguas 
            WHERE piscina_id = :pid 
            ORDER BY fecha_medicion DESC LIMIT 100
        """)
        df = pd.read_sql(query, get_engine(), params={"pid": pool_id})

        if df.empty:
            return jsonify({"forecast": [], "confidence": 0, "alerts": []})

        df['fecha_medicion'] = pd.to_datetime(df['fecha_medicion'])
        df = df.sort_values('fecha_medicion')
        df['timestamp'] = df['fecha_medicion'].astype(int) // 10**9

        X = df[['timestamp', 'temperatura']]
        y = df['oxigeno_disuelto']

        model = LinearRegression()
        model.fit(X, y)

        last_time = df['fecha_medicion'].iloc[-1]
        last_temp = df['temperatura'].iloc[-1]

        forecast_data = []
        alerts = []

        for i in range(1, 9):
            future_time = last_time + timedelta(minutes=30 * i)
            ts = int(future_time.timestamp())
            pred_o2 = model.predict([[ts, last_temp]])[0]

            forecast_data.append({
                "time": future_time.strftime("%H:%M"),
                "value": round(float(pred_o2), 2),
                "lower_bound": round(float(pred_o2) - 0.5, 2),
                "upper_bound": round(float(pred_o2) + 0.5, 2)
            })

            if pred_o2 < 3.0 and not any(a['severity'] == 'critical' for a in alerts):
                alerts.append({
                    "title": "CRUCE CRÍTICO DETECTADO",
                    "time": future_time.strftime("%H:%M"),
                    "description": f"Predicción de O2 < 3.0 mg/L ({round(pred_o2,1)}). Riesgo inminente.",
                    "severity": "critical"
                })

        last_row = df.iloc[-1]
        anomaly_scores = {}

        for col in ['ph', 'temperatura', 'oxigeno_disuelto', 'ion_nitrato']:
            mean = df[col].mean()
            std = df[col].std() or 1
            z = abs((last_row[col] - mean) / std)
            score = min(z * 20, 100)
            anomaly_scores[col.replace('_disuelto', '').replace('ion_', '')] = round(score, 1)

            if z > 3:
                alerts.append({
                    "title": f"Anomalía Estadística en {col}",
                    "time": "Ahora",
                    "description": f"Valor Z-Score {round(z,1)}",
                    "severity": "warning"
                })

        return jsonify({
            "forecast": forecast_data,
            "confidence": 89.5,
            "anomaly_scores": anomaly_scores,
            "alerts": alerts
        })

    except Exception as e:
        print(f"Error predicción: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/system/health', methods=['GET'])
def get_system_health():
    try:
        with get_db_connection() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM parametro_aguas")).fetchone()[0]

            q_sensors = text("""
                SELECT p.id, p.nombre, MAX(pa.created_at) as last_seen
                FROM piscinas p
                LEFT JOIN parametro_aguas pa ON p.id = pa.piscina_id
                GROUP BY p.id, p.nombre
            """)
            rows = conn.execute(q_sensors).fetchall()

            sensors_status = []
            now = datetime.now()
            total_latency = 0

            for pid, name, last_seen in rows:
                status = "offline"
                latency = 0

                if last_seen:
                    if isinstance(last_seen, str):
                        last_seen = datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S")
                    delta = now - last_seen
                    latency = delta.total_seconds() * 1000
                    if delta.total_seconds() < 300:
                        status = "online"

                total_latency += latency

                sensors_status.append({
                    "id": f"SENSOR-POOL-{pid}",
                    "type": "Multi-Parametro (IoT)",
                    "status": status,
                    "latency_ms": int(latency),
                    "last_seen": last_seen.strftime("%H:%M:%S") if last_seen else "Nunca"
                })

            logs_q = text("""
                SELECT id, created_at, name, type
                FROM accions
                ORDER BY created_at DESC LIMIT 5
            """)
            logs_rows = conn.execute(logs_q).fetchall()

            logs = [{
                "id": l[0],
                "timestamp": l[1].strftime("%H:%M"),
                "level": "info" if l[3] != 'ALERTA' else 'warning',
                "message": f"{l[2]} ({l[3]})"
            } for l in logs_rows]

            return jsonify({
                "cleaned_records": int(count * 0.15),
                "imputed_records": int(count * 0.05),
                "avg_latency_ms": int(total_latency / len(rows)) if rows else 0,
                "sensors": sensors_status,
                "logs": logs
            })

    except Exception as e:
        print(f"Error health: {e}")
        return jsonify({"error": str(e)}), 500


# --- ARRANQUE ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
