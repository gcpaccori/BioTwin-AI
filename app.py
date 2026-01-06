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
from sqlalchemy.exc import OperationalError

from sklearn.linear_model import LinearRegression

# =========================================================
# CONFIGURACIÓN FLASK
# =========================================================
app = Flask(__name__)
CORS(app)

# =========================================================
# CONFIGURACIÓN BASE DE DATOS
# =========================================================
DB_HOST = os.getenv("DB_HOST", "37.60.226.53")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "sismapiscis2025")
DB_NAME = os.getenv("DB_NAME", "sismapiscis")
DB_PORT = int(os.getenv("DB_PORT", "3306"))

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

_engine = None


def get_engine():
    """
    Crea el engine SOLO cuando se necesita.
    Evita timeouts en gunicorn / Leapcell.
    """
    global _engine
    if _engine is None:
        print("🚀 Inicializando conexión MySQL (lazy)")
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 5},
        )
    return _engine


def get_db_connection():
    return get_engine().connect()


# =========================================================
# ENDPOINT ROOT – TEST DE VIDA + DB
# =========================================================
@app.route("/", methods=["GET"])
def root_health():
    """
    Endpoint obligatorio para Leapcell.
    Verifica si la app vive y si MySQL responde.
    """
    try:
        with get_db_connection() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({
            "status": "ok",
            "service": "BioTwin AI Backend",
            "database": "connected"
        })
    except Exception as e:
        return jsonify({
            "status": "warning",
            "service": "BioTwin AI Backend",
            "database": "not_connected",
            "error": str(e)
        }), 503


# =========================================================
# UTILIDADES
# =========================================================
def calcular_color_agua(nitrato):
    if nitrato <= 10:
        return "#0EA5E9"
    if nitrato <= 40:
        return "#22C55E"
    return "#854D0E"


# =========================================================
# API V1 – BIOFLOC STATUS
# =========================================================
@app.route("/api/v1/biofloc/status/<int:pool_id>", methods=["GET"])
def get_biofloc_status(pool_id):
    try:
        with get_db_connection() as conn:
            q = text("""
                SELECT ion_nitrato, oxigeno_disuelto, ph, temperatura
                FROM parametro_aguas
                WHERE piscina_id = :pid
                ORDER BY fecha_medicion DESC
                LIMIT 1
            """)
            row = conn.execute(q, {"pid": pool_id}).fetchone()

            if not row:
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

            nitrato, o2, ph, temp = row

            target_nitrato = 10.0
            carbon_demand = 0
            carbon_amount_gr = 0

            if nitrato > target_nitrato:
                delta_n = nitrato - target_nitrato
                carbon_demand = delta_n * 15
                carbon_amount_gr = round(carbon_demand * 20, 2)

            dosing_locked = o2 < 4.0
            estado_critico = nitrato > 50 or o2 < 3.0

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
        return jsonify({"error": str(e)}), 500


# =========================================================
# API V1 – DIGITAL TWIN CONFIG
# =========================================================
@app.route("/api/v1/digital-twin/config/<int:pool_id>", methods=["GET"])
def get_digital_twin_config(pool_id):
    try:
        with get_db_connection() as conn:
            pool = conn.execute(
                text("SELECT largo, ancho, profundidad FROM piscinas WHERE id=:id"),
                {"id": pool_id}
            ).fetchone()

            water = conn.execute(
                text("""
                    SELECT ion_nitrato
                    FROM parametro_aguas
                    WHERE piscina_id=:id
                    ORDER BY id DESC LIMIT 1
                """),
                {"id": pool_id}
            ).fetchone()

            bio = conn.execute(
                text("""
                    SELECT cantidad_muestreo, peso_promedio
                    FROM biometrias
                    WHERE piscina_id=:id
                    ORDER BY fecha_registro DESC LIMIT 1
                """),
                {"id": pool_id}
            ).fetchone()

            l, w, d = pool if pool else (10, 5, 1.2)
            nitrato = water[0] if water else 0
            fish_count, avg_weight = bio if bio else (1000, 10)

            return jsonify({
                "dimensions": {"l": float(l), "w": float(w), "d": float(d)},
                "water_quality": {
                    "color_hex": calcular_color_agua(nitrato),
                    "turbidity_factor": min(nitrato / 100, 1.0)
                },
                "biomass": {
                    "fish_count": int(fish_count),
                    "avg_weight": float(avg_weight)
                }
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# API V1 – OXYGEN PREDICTION
# =========================================================
@app.route("/api/v1/predict/oxygen/<int:pool_id>", methods=["GET"])
def get_oxygen_prediction(pool_id):
    try:
        df = pd.read_sql(
            text("""
                SELECT fecha_medicion, oxigeno_disuelto, temperatura, ph, ion_nitrato
                FROM parametro_aguas
                WHERE piscina_id=:id
                ORDER BY fecha_medicion DESC
                LIMIT 100
            """),
            get_engine(),
            params={"id": pool_id}
        )

        if df.empty:
            return jsonify({"forecast": [], "confidence": 0, "alerts": []})

        df["fecha_medicion"] = pd.to_datetime(df["fecha_medicion"])
        df = df.sort_values("fecha_medicion")
        df["timestamp"] = df["fecha_medicion"].astype("int64") // 10**9

        model = LinearRegression()
        model.fit(df[["timestamp", "temperatura"]], df["oxigeno_disuelto"])

        last_time = df["fecha_medicion"].iloc[-1]
        last_temp = df["temperatura"].iloc[-1]

        forecast, alerts = [], []

        for i in range(1, 9):
            ft = last_time + timedelta(minutes=30 * i)
            pred = model.predict([[int(ft.timestamp()), last_temp]])[0]

            forecast.append({
                "time": ft.strftime("%H:%M"),
                "value": round(pred, 2),
                "lower_bound": round(pred - 0.5, 2),
                "upper_bound": round(pred + 0.5, 2),
            })

            if pred < 3 and not alerts:
                alerts.append({
                    "title": "CRUCE CRÍTICO DETECTADO",
                    "time": ft.strftime("%H:%M"),
                    "description": "Riesgo de hipoxia inminente",
                    "severity": "critical"
                })

        return jsonify({
            "forecast": forecast,
            "confidence": 89.5,
            "alerts": alerts
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# MAIN (solo para local)
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
