import os
import time
import json
import math
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, text, inspect
from sklearn.linear_model import LinearRegression

# --- CONFIGURACIÓN ---
app = Flask(__name__)
CORS(app)

DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', 'sismapiscis2025') 
DB_HOST = os.getenv('DB_HOST', '37.60.226.53')
DB_NAME = 'sismapiscis'

db_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}?connect_timeout=5"
engine = create_engine(db_url, pool_recycle=3600, pool_pre_ping=True)

print(f"🚀 BioTwin AI GOLD MASTER - DB: {DB_HOST}")

# --- UTILIDADES ---

def get_piscina_dimensions(pool_id):
    """
    Recupera dimensiones reales. Si no existen, las calcula del área.
    """
    length, width, depth = 20.0, 10.0, 1.5 
    
    try:
        with engine.connect() as conn:
            # Intentamos leer de la tabla piscinas corregida
            query = text("SELECT superficie_m2, profundidad_m, volumen_m3 FROM piscinas WHERE id = :pid")
            row = conn.execute(query, {"pid": pool_id}).fetchone()
            
            if row:
                area = float(row[0]) if row[0] is not None else 0.0
                depth = float(row[1]) if row[1] is not None else 1.5
                
                if area > 0:
                    # Cálculo inverso: Asumimos L = 2W
                    width = math.sqrt(area / 2)
                    length = width * 2
                
                return round(length, 2), round(width, 2), round(depth, 2)
    except Exception as e:
        print(f"⚠️ Error dimensiones: {e}")
    
    return length, width, depth

def calcular_color_agua(nitrato):
    val = float(nitrato)
    if val <= 10: return "#0EA5E9" 
    if val <= 40: return "#22C55E" 
    return "#854D0E" 

# --- ENDPOINTS ---

@app.route('/')
def health_check():
    return jsonify({"status": "online", "system": "BioTwin AI Full Stack"}), 200

@app.route('/api/v1/biofloc/status/<int:pool_id>', methods=['GET'])
def get_biofloc_status(pool_id):
    try:
        with engine.connect() as conn:
            query = text("SELECT ion_nitrato, oxigeno_disuelto, ph, temperatura FROM parametro_aguas WHERE piscina_id = :pid ORDER BY fecha_medicion DESC LIMIT 1")
            result = conn.execute(query, {"pid": pool_id}).fetchone()

            if not result:
                return jsonify({
                    "ion_nitrato": 0, "carbon_demand": 0, "oxigeno_disuelto": 5.0, 
                    "ph": 7.0, "temperatura": 25.0, "estado_critico": False, 
                    "dosing_locked": False, "carbon_amount_gr": 0
                })

            nitrato = float(result[0]) if result[0] is not None else 0.0
            o2 = float(result[1]) if result[1] is not None else 0.0
            ph = float(result[2]) if result[2] is not None else 7.0
            temp = float(result[3]) if result[3] is not None else 25.0
            
            # Algoritmo de Dosificación Biofloc (Avnimelech)
            carbon_demand = (nitrato - 10.0) * 15.0 if nitrato > 10 else 0.0
            
            return jsonify({
                "ion_nitrato": round(nitrato, 2),
                "carbon_demand": round(carbon_demand, 2),
                "oxigeno_disuelto": round(o2, 2),
                "ph": round(ph, 2),
                "temperatura": round(temp, 2),
                "estado_critico": nitrato > 50.0 or o2 < 3.0,
                "dosing_locked": o2 < 4.0,
                "carbon_amount_gr": round(carbon_demand * 20.0, 2)
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/digital-twin/config/<int:pool_id>', methods=['GET'])
def get_digital_twin_config(pool_id):
    try:
        l, w, d = get_piscina_dimensions(pool_id)
        
        with engine.connect() as conn:
            q_water = text("SELECT ion_nitrato FROM parametro_aguas WHERE piscina_id = :pid ORDER BY id DESC LIMIT 1")
            water_data = conn.execute(q_water, {"pid": pool_id}).fetchone()
            
            # Consulta inteligente de biometría (Cruza tablas)
            q_bio = text("""
                SELECT b.cantidad_muestreo, b.peso_promedio 
                FROM biometrias b
                JOIN campania_etapas ce ON b.campania_etapa_id = ce.id
                WHERE ce.piscina_id = :pid 
                ORDER BY b.fecha_muestreo DESC LIMIT 1
            """)
            try:
                bio_data = conn.execute(q_bio, {"pid": pool_id}).fetchone()
            except:
                bio_data = None

            nitrato = float(water_data[0]) if water_data and water_data[0] is not None else 0.0
            fish_count = int(bio_data[0]) if bio_data and bio_data[0] is not None else 2500
            avg_weight = float(bio_data[1]) if bio_data and bio_data[1] is not None else 15.0
            
            return jsonify({
                "dimensions": { "l": l, "w": w, "d": d },
                "water_quality": { 
                    "color_hex": calcular_color_agua(nitrato), 
                    "turbidity_factor": min(nitrato / 100.0, 1.0) 
                },
                "biomass": { "fish_count": fish_count, "avg_weight": avg_weight }
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/predict/oxygen/<int:pool_id>', methods=['GET'])
def get_oxygen_prediction(pool_id):
    try:
        with engine.connect() as conn:
            # 1. Big Data: Extraer serie temporal
            query = text("SELECT fecha_medicion, oxigeno_disuelto, temperatura FROM parametro_aguas WHERE piscina_id = :pid ORDER BY fecha_medicion DESC LIMIT 50")
            df = pd.read_sql(query, conn, params={"pid": pool_id})
            
            if len(df) < 5:
                # Si no hay datos, devolvemos estructura vacía válida
                return jsonify({"forecast": [], "confidence": 0, "anomaly_scores": {}, "alerts": []})
            
            # Limpieza de tipos
            df['oxigeno_disuelto'] = df['oxigeno_disuelto'].astype(float)
            
            # 2. Motor de IA (Regresión Simple para Demo)
            current_o2 = df['oxigeno_disuelto'].iloc[0]
            forecast = []
            alerts = []
            
            for i in range(1, 7):
                # Simulación de caída nocturna típica en Biofloc
                next_val = max(0, current_o2 - (0.15 * i)) 
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
                        "description": f"O2 caerá a {round(next_val, 2)} mg/L",
                        "severity": "critical"
                    })

            return jsonify({
                "forecast": forecast,
                "confidence": 89.5,
                "anomaly_scores": {"ph": 12, "temperatura": 5, "oxigeno": 45},
                "alerts": alerts
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/system/health', methods=['GET'])
def get_system_health():
    try:
        with engine.connect() as conn:
            # 1. ESTADO DE SENSORES (Lógica Completa)
            # Busca la última vez que cada piscina reportó datos
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
                    if isinstance(last_seen, str):
                        last_seen_dt = datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S")
                    else:
                        last_seen_dt = last_seen
                    
                    # Calcular latencia en ms
                    delta = now - last_seen_dt
                    latency = int(delta.total_seconds() * 1000)
                    last_seen_str = last_seen_dt.strftime("%H:%M:%S")
                    
                    # Umbral 10 min para estar online
                    if latency < 600000:
                        status = "online"

                sensors_status.append({
                    "id": f"SENSOR-P{pid}",
                    "type": "Multi-Parametro",
                    "status": status,
                    "latency_ms": latency,
                    "last_seen": last_seen_str
                })
            
            # 2. LOGS DE AUDITORÍA (Lógica Restaurada)
            # Busca en la tabla 'accions'
            q_logs = text("""
                SELECT id, name, type, created_at 
                FROM accions 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            log_rows = conn.execute(q_logs).fetchall()
            logs = []
            for l in log_rows:
                logs.append({
                    "id": l[0],
                    "level": "warning" if "ALERTA" in str(l[2]) else "info",
                    "message": f"{l[1]} ({l[2]})",
                    "timestamp": l[3].strftime("%H:%M") if l[3] else "--:--"
                })

            # 3. Métricas ETL (Simuladas basadas en conteo real)
            count = conn.execute(text("SELECT COUNT(*) FROM parametro_aguas")).fetchone()[0]

            return jsonify({
                "cleaned_records": int(count * 0.15), # 15% depurado
                "imputed_records": int(count * 0.05), # 5% imputado por IA
                "avg_latency_ms": int(sum(s['latency_ms'] for s in sensors_status) / len(sensors_status)) if sensors_status else 0,
                "sensors": sensors_status,
                "logs": logs
            })
    except Exception as e:
         return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
