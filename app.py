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

print(f"🚀 BioTwin AI v4.0 (Esquema Corregido) - DB: {DB_HOST}")

# --- UTILIDADES ---

def get_piscina_dimensions(pool_id):
    """
    Calcula dimensiones 3D basadas en 'superficie_m2' y 'profundidad_m'.
    """
    # Valores por defecto
    length, width, depth = 20.0, 10.0, 1.5 
    
    try:
        with engine.connect() as conn:
            # CONSULTA EXACTA PARA TU ESTRUCTURA
            query = text("SELECT superficie_m2, profundidad_m, volumen_m3 FROM piscinas WHERE id = :pid")
            row = conn.execute(query, {"pid": pool_id}).fetchone()
            
            if row:
                # Convertimos Decimal a float de Python
                area = float(row[0]) if row[0] is not None else 0.0
                depth = float(row[1]) if row[1] is not None else 1.5
                
                # Si el área es 0 o null, usamos default
                if area > 0:
                    # Matemáticas: Asumimos piscina rectangular con proporción 2:1
                    # Area = L * W
                    # L = 2W  -->  Area = 2W * W = 2W^2
                    # W = raiz(Area / 2)
                    width = math.sqrt(area / 2)
                    length = width * 2
                
                return round(length, 2), round(width, 2), round(depth, 2)

    except Exception as e:
        print(f"⚠️ Error calculando dimensiones: {e}")
    
    return length, width, depth

def calcular_color_agua(nitrato):
    val = float(nitrato)
    if val <= 10: return "#0EA5E9" 
    if val <= 40: return "#22C55E" 
    return "#854D0E" 

# --- ENDPOINTS ---

@app.route('/')
def health_check():
    return jsonify({"status": "online", "mode": "Production Schema Fixed"}), 200

@app.route('/api/v1/biofloc/status/<int:pool_id>', methods=['GET'])
def get_biofloc_status(pool_id):
    try:
        with engine.connect() as conn:
            query = text("SELECT ion_nitrato, oxigeno_disuelto, ph, temperatura FROM parametro_aguas WHERE piscina_id = :pid ORDER BY fecha_medicion DESC LIMIT 1")
            result = conn.execute(query, {"pid": pool_id}).fetchone()

            if not result:
                return jsonify({
                    "ion_nitrato": 0, "carbon_demand": 0, "oxigeno_disuelto": 5.0, 
                    "estado_critico": False, "dosing_locked": False, "carbon_amount_gr": 0
                })

            # Conversión segura a float
            nitrato = float(result[0]) if result[0] is not None else 0.0
            o2 = float(result[1]) if result[1] is not None else 0.0
            ph = float(result[2]) if result[2] is not None else 7.0
            temp = float(result[3]) if result[3] is not None else 25.0
            
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
        # 1. Dimensiones calculadas desde superficie_m2
        l, w, d = get_piscina_dimensions(pool_id)
        
        with engine.connect() as conn:
            # 2. Calidad Agua
            q_water = text("SELECT ion_nitrato FROM parametro_aguas WHERE piscina_id = :pid ORDER BY id DESC LIMIT 1")
            water_data = conn.execute(q_water, {"pid": pool_id}).fetchone()
            
            # 3. Biomasa (Join corregido)
            # Intenta buscar biometría vinculada a la campaña de la piscina
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
        # Fallback para que el frontend 3D nunca rompa
        return jsonify({
            "dimensions": { "l": 20.0, "w": 10.0, "d": 1.5 },
            "water_quality": { "color_hex": "#0EA5E9", "turbidity_factor": 0.1 },
            "biomass": { "fish_count": 1000, "avg_weight": 10.0 },
            "debug": str(e)
        })

@app.route('/api/v1/predict/oxygen/<int:pool_id>', methods=['GET'])
def get_oxygen_prediction(pool_id):
    # Lógica de predicción simulada para demo (se mantiene igual)
    return jsonify({
        "forecast": [], "confidence": 85.0, "anomaly_scores": {}, "alerts": []
    })

@app.route('/api/v1/system/health', methods=['GET'])
def get_system_health():
    # Lógica de salud (se mantiene igual)
    return jsonify({
        "cleaned_records": 350, "imputed_records": 116, 
        "avg_latency_ms": 45, "sensors": [], "logs": []
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
