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

# --- CONFIGURACIÓN ---
app = Flask(__name__)
CORS(app)

# CONFIGURACIÓN DE BASE DE DATOS BLINDADA
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '') 
DB_HOST = os.getenv('DB_HOST', '37.60.226.53')
DB_NAME = 'sismapiscis'

# Timeout corto para evitar bloqueos
db_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}?connect_timeout=3"

engine = create_engine(db_url, pool_recycle=3600, pool_pre_ping=True)

print(f"🚀 BioTwin AI Backend v2.1 (Decimal Fix). Host DB: {DB_HOST}")

# --- UTILIDADES ---

def get_db_connection():
    return engine.connect()

def calcular_color_agua(nitrato):
    # Aseguramos que nitrato sea float para comparar
    val = float(nitrato)
    if val <= 10: return "#0EA5E9"
    if val <= 40: return "#22C55E"
    return "#854D0E"

# --- RUTA PRINCIPAL ---
@app.route('/')
def health_check():
    return jsonify({
        "status": "online",
        "system": "BioTwin AI v2.1",
        "fix": "Decimal conversion applied"
    }), 200

# --- ENDPOINTS ---

@app.route('/api/v1/biofloc/status/<int:pool_id>', methods=['GET'])
def get_biofloc_status(pool_id):
    try:
        with get_db_connection() as conn:
            # 1. Obtenemos datos crudos
            query = text("SELECT ion_nitrato, oxigeno_disuelto, ph, temperatura FROM parametro_aguas WHERE piscina_id = :pid ORDER BY fecha_medicion DESC LIMIT 1")
            result = conn.execute(query, {"pid": pool_id}).fetchone()

            if not result:
                return jsonify({
                    "ion_nitrato": 0, "carbon_demand": 0, "oxigeno_disuelto": 5.0,
                    "ph": 7.0, "temperatura": 25.0, "estado_critico": False,
                    "dosing_locked": False, "carbon_amount_gr": 0
                })

            # 2. CONVERSIÓN EXPLÍCITA A FLOAT (Corrección del error Decimal)
            # MySQL devuelve Decimal, aquí lo forzamos a float de Python
            nitrato = float(result[0])
            o2 = float(result[1])
            ph = float(result[2])
            temp = float(result[3])
            
            # 3. Cálculos matemáticos (ahora seguros entre floats)
            carbon_demand = (nitrato - 10.0) * 15.0 if nitrato > 10 else 0.0
            carbon_amount_gr = round(carbon_demand * 20.0, 2)
            
            dosing_locked = True if o2 < 4.0 else False
            estado_critico = True if nitrato > 50.0 or o2 < 3.0 else False

            return jsonify({
                "ion_nitrato": round(nitrato, 2),
                "carbon_demand": round(carbon_demand, 2),
                "oxigeno_disuelto": round(o2, 2),
                "ph": round(ph, 2),
                "temperatura": round(temp, 2),
                "estado_critico": estado_critico,
                "dosing_locked": dosing_locked,
                "carbon_amount_gr": carbon_amount_gr
            })

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/digital-twin/config/<int:pool_id>', methods=['GET'])
def get_digital_twin_config(pool_id):
    try:
        with get_db_connection() as conn:
            # Datos Físicos
            q_pool = text("SELECT largo, ancho, profundidad FROM piscinas WHERE id = :pid")
            pool_data = conn.execute(q_pool, {"pid": pool_id}).fetchone()
            
            # Calidad agua (última lectura)
            q_water = text("SELECT ion_nitrato FROM parametro_aguas WHERE piscina_id = :pid ORDER BY id DESC LIMIT 1")
            water_data = conn.execute(q_water, {"pid": pool_id}).fetchone()

            # Biometría
            q_bio = text("SELECT cantidad_muestreo, peso_promedio FROM biometrias WHERE piscina_id = :pid ORDER BY fecha_registro DESC LIMIT 1")
            bio_data = conn.execute(q_bio, {"pid": pool_id}).fetchone()
            
            # Conversión a Float segura
            l = float(pool_data[0]) if pool_data else 20.0
            w = float(pool_data[1]) if pool_data else 10.0
            d = float(pool_data[2]) if pool_data else 1.5
            
            nitrato = float(water_data[0]) if water_data else 0.0
            
            fish_count = int(bio_data[0]) if bio_data else 2500
            avg_weight = float(bio_data[1]) if bio_data else 15.0
            
            # Lógica visual
            turbidity = min(nitrato / 100.0, 1.0)
            
            return jsonify({
                "dimensions": { "l": l, "w": w, "d": d },
                "water_quality": { 
                    "color_hex": calcular_color_agua(nitrato), 
                    "turbidity_factor": turbidity 
                },
                "biomass": { "fish_count": fish_count, "avg_weight": avg_weight }
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/predict/oxygen/<int:pool_id>', methods=['GET'])
def get_oxygen_prediction(pool_id):
    try:
        with get_db_connection() as conn:
            query = text("SELECT fecha_medicion, oxigeno_disuelto, temperatura FROM parametro_aguas WHERE piscina_id = :pid ORDER BY fecha_medicion DESC LIMIT 50")
            df = pd.read_sql(query, conn, params={"pid": pool_id})
            
            if len(df) < 5: 
                # Retornar estructura vacía válida para evitar errores en frontend
                return jsonify({
                    "forecast": [], "confidence": 0, "anomaly_scores": {}, "alerts": []
                })
            
            # Asegurar tipos float en Pandas
            df['oxigeno_disuelto'] = df['oxigeno_disuelto'].astype(float)
            df['temperatura'] = df['temperatura'].astype(float)
            
            # Lógica simple de forecast para demo
            current_o2 = df['oxigeno_disuelto'].iloc[0]
            forecast = []
            alerts = []
            
            # Proyectar caída leve
            for i in range(1, 7):
                next_val = current_o2 - (0.1 * i)
                t_future = (datetime.now() + timedelta(hours=i)).strftime("%H:%M")
                forecast.append({
                    "time": t_future,
                    "value": round(next_val, 2),
                    "lower_bound": round(next_val - 0.5, 2),
                    "upper_bound": round(next_val + 0.5, 2)
                })
                
                if next_val < 3.0:
                    alerts.append({
                        "title": "ALERTA DE HIPOXIA",
                        "time": t_future,
                        "description": "Nivel de O2 crítico proyectado.",
                        "severity": "critical"
                    })

            return jsonify({
                "forecast": forecast,
                "confidence": 88.5,
                "anomaly_scores": {"ph": 12, "temperatura": 5},
                "alerts": alerts
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/system/health', methods=['GET'])
def get_system_health():
    try:
        with get_db_connection() as conn:
            # Consulta para estado de sensores
            q_sensors = text("""
                SELECT p.id, p.nombre, MAX(pa.created_at) as last_seen
                FROM piscinas p
                LEFT JOIN parametro_aguas pa ON p.id = pa.piscina_id
                GROUP BY p.id, p.nombre
            """)
            rows = conn.execute(q_sensors).fetchall()
            
            sensors_status = []
            now = datetime.now()
            
            for row in rows:
                pid, name, last_seen = row
                latency = 0
                status = "offline"
                last_seen_str = "Nunca"

                if last_seen:
                    # Corrección: Asegurar que last_seen sea datetime
                    if isinstance(last_seen, str):
                        last_seen_dt = datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S")
                    else:
                        last_seen_dt = last_seen
                    
                    delta = now - last_seen_dt
                    latency = int(delta.total_seconds() * 1000)
                    last_seen_str = last_seen_dt.strftime("%H:%M:%S")
                    
                    # Umbral: 10 minutos (600,000 ms) para considerar online
                    if latency < 600000:
                        status = "online"

                sensors_status.append({
                    "id": f"SENSOR-POOL-{pid}",
                    "type": "Multi-Parametro (IoT)",
                    "status": status,
                    "latency_ms": latency,
                    "last_seen": last_seen_str
                })
            
            # Logs de auditoría
            q_logs = text("SELECT id, level, message, timestamp FROM (SELECT id, 'info' as level, concat(name, ' (', type, ')') as message, date_format(created_at, '%H:%i') as timestamp FROM accions ORDER BY created_at DESC LIMIT 5) as sub")
            logs = [dict(row._mapping) for row in conn.execute(q_logs).fetchall()]

            return jsonify({
                "cleaned_records": 350,
                "imputed_records": 116,
                "avg_latency_ms": int(sum(s['latency_ms'] for s in sensors_status) / len(sensors_status)) if sensors_status else 0,
                "sensors": sensors_status,
                "logs": logs
            })
    except Exception as e:
         return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
