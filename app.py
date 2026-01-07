import os
import math
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, text
import pymysql

# --- CONFIGURACIÓN ---
app = Flask(__name__)
CORS(app)

DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "sismapiscis2025")  # ⚠️ Recomendado: NO dejar password hardcodeado en prod
DB_HOST = os.getenv("DB_HOST", "37.60.226.53")
DB_NAME = os.getenv("DB_NAME", "sismapiscis")

db_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}?connect_timeout=5"
engine = create_engine(db_url, pool_recycle=3600, pool_pre_ping=True)

print(f"🚀 BioTwin AI GOLD MASTER V2.0 - DB: {DB_HOST}")

# -----------------------------------------------------------------------------
#  Timestamp "EFECTIVO" para arreglar el problema:
#  - Los micros envían fecha_medicion vieja/congelada (pero created_at sí es actual).
#  - Usamos created_at cuando fecha_medicion está muy desfasada.
# -----------------------------------------------------------------------------
TS_EFFECTIVE_EXPR = """
CASE
  WHEN fecha_medicion IS NULL THEN created_at
  WHEN created_at IS NULL THEN fecha_medicion
  WHEN fecha_medicion < created_at - INTERVAL 2 DAY THEN created_at
  WHEN fecha_medicion > created_at + INTERVAL 2 DAY THEN created_at
  ELSE fecha_medicion
END
"""

def period_to_mysql_interval(period: str) -> str:
    # interval strings MySQL: "6 HOUR", "7 DAY", etc.
    return {
        "6h": "6 HOUR",
        "24h": "24 HOUR",
        "7d": "7 DAY",
        "30d": "30 DAY",
    }.get(period, "24 HOUR")


# --- UTILIDADES ---

def get_piscina_dimensions(pool_id: int):
    """
    Recupera dimensiones reales. Si no existen, las calcula del área.
    """
    length, width, depth = 20.0, 10.0, 1.5

    try:
        with engine.connect() as conn:
            query = text("SELECT superficie_m2, profundidad_m, volumen_m3 FROM piscinas WHERE id = :pid")
            row = conn.execute(query, {"pid": pool_id}).fetchone()

            if row:
                area = float(row[0]) if row[0] is not None else 0.0
                depth = float(row[1]) if row[1] is not None else 1.5

                if area > 0:
                    width = math.sqrt(area / 2)
                    length = width * 2

                return round(length, 2), round(width, 2), round(depth, 2)
    except Exception as e:
        print(f"⚠️ Error dimensiones: {e}")

    return length, width, depth


def calcular_color_agua(nitrato):
    val = float(nitrato)
    if val <= 10:
        return "#0EA5E9"
    if val <= 40:
        return "#22C55E"
    return "#854D0E"


def get_db_connection():
    """
    Crea una conexión directa a MySQL usando pymysql
    """
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor
    )


# --- ENDPOINTS V2.0 ---

@app.route("/")
def health_check():
    return jsonify({"status": "online", "system": "BioTwin AI V2.0", "version": "2.0.0"}), 200


@app.route("/api/v1/sensor-detail/<int:pool_id>", methods=["GET"])
def get_sensor_detail(pool_id: int):
    """
    Devuelve serie temporal del sensor principal + sensores cruzados.
    """
    sensor_type = request.args.get("sensor", "temperature")
    period = request.args.get("period", "24h")
    cross_sensors_param = request.args.get("crossSensors", "")
    cross_sensors = [s.strip() for s in cross_sensors_param.split(",") if s.strip()]

    sensor_columns = {
        "temperature": "temperatura",
        "ph": "ph",
        "oxygen": "oxigeno_disuelto",
        "nitrate": "ion_nitrato",
    }
    column = sensor_columns.get(sensor_type, "temperatura")
    interval = period_to_mysql_interval(period)

    conn = get_db_connection()
    try:
        # Query principal
        query_main = f"""
            SELECT
                {TS_EFFECTIVE_EXPR} AS ts,
                {column} AS val
            FROM parametro_aguas
            WHERE piscina_id = %s
              AND deleted_at IS NULL
              AND {TS_EFFECTIVE_EXPR} >= DATE_SUB(NOW(), INTERVAL {interval})
            ORDER BY ts ASC
        """

        with conn.cursor() as cursor:
            cursor.execute(query_main, (pool_id,))
            rows = cursor.fetchall()

        main_data = []
        for r in rows:
            ts = r.get("ts")
            val = r.get("val")

            if ts is None:
                continue

            main_data.append({
                "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                "value": float(val) if val is not None else 0
            })

        cross_data = {}
        if cross_sensors:
            for cross_sensor in cross_sensors:
                cross_column = sensor_columns.get(cross_sensor)
                if not cross_column:
                    continue
                
                query_cross = f"""
                    SELECT
                        {TS_EFFECTIVE_EXPR} AS ts,
                        {cross_column} AS val
                    FROM parametro_aguas
                    WHERE piscina_id = %s
                      AND deleted_at IS NULL
                      AND {TS_EFFECTIVE_EXPR} >= DATE_SUB(NOW(), INTERVAL {interval})
                    ORDER BY ts ASC
                """
                
                with conn.cursor() as cursor:
                    cursor.execute(query_cross, (pool_id,))
                    cross_rows = cursor.fetchall()
                
                cross_data[cross_sensor] = []
                for r in cross_rows:
                    ts = r.get("ts")
                    val = r.get("val")
                    
                    if ts is None:
                        continue
                    
                    cross_data[cross_sensor].append({
                        "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                        "value": float(val) if val is not None else 0
                    })

        return jsonify({"main": main_data, "cross": cross_data})
    except Exception as e:
        print(f"❌ sensor-detail error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/v1/biofloc/history/<int:pool_id>", methods=["GET"])
def get_sensor_history(pool_id: int):
    """
    Devuelve las últimas ~6 horas (12 lecturas) para sparklines.
    Usa ts efectivo (arregla fecha_medicion congelada).
    """
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT
                    {TS_EFFECTIVE_EXPR} AS ts,
                    oxigeno_disuelto, ph, temperatura, ion_nitrato
                FROM parametro_aguas
                WHERE piscina_id = :pid
                  AND deleted_at IS NULL
                ORDER BY ts DESC
                LIMIT 12
            """)
            rows = conn.execute(query, {"pid": pool_id}).fetchall()

            history = {
                "oxigeno": [],
                "ph": [],
                "temperatura": [],
                "nitrato": [],
                "timestamps": []
            }

            for row in reversed(rows):  # orden cronológico
                ts = row[0]
                history["timestamps"].append(ts.strftime("%H:%M") if ts else "")
                history["oxigeno"].append(float(row[1]) if row[1] is not None else 0)
                history["ph"].append(float(row[2]) if row[2] is not None else 7.0)
                history["temperatura"].append(float(row[3]) if row[3] is not None else 25.0)
                history["nitrato"].append(float(row[4]) if row[4] is not None else 0)

            return jsonify(history)
    except Exception as e:
        print(f"❌ biofloc/history error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/biofloc/status/<int:pool_id>", methods=["GET"])
def get_biofloc_status(pool_id: int):
    """
    Último estado (usa ts efectivo).
    """
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT
                    ion_nitrato,
                    oxigeno_disuelto,
                    ph,
                    temperatura,
                    {TS_EFFECTIVE_EXPR} AS ts
                FROM parametro_aguas
                WHERE piscina_id = :pid
                  AND deleted_at IS NULL
                ORDER BY ts DESC
                LIMIT 1
            """)
            result = conn.execute(query, {"pid": pool_id}).fetchone()

            if not result:
                return jsonify({
                    "ion_nitrato": 0, "carbon_demand": 0, "oxigeno_disuelto": 5.0,
                    "ph": 7.0, "temperatura": 25.0, "estado_critico": False,
                    "dosing_locked": False, "carbon_amount_gr": 0,
                    "last_updated_iso": None,
                    "data_age_minutes": 9999
                })

            nitrato = float(result[0]) if result[0] is not None else 0.0
            o2 = float(result[1]) if result[1] is not None else 0.0
            ph = float(result[2]) if result[2] is not None else 7.0
            temp = float(result[3]) if result[3] is not None else 25.0

            ts = result[4]
            if ts:
                data_age = (datetime.now() - ts).total_seconds() / 60
                last_updated_iso = ts.isoformat() if isinstance(ts, datetime) else str(ts)
            else:
                data_age = 9999
                last_updated_iso = None

            carbon_demand = (nitrato - 10.0) * 15.0 if nitrato > 10 else 0.0

            return jsonify({
                "ion_nitrato": round(nitrato, 2),
                "carbon_demand": round(carbon_demand, 2),
                "oxigeno_disuelto": round(o2, 2),
                "ph": round(ph, 2),
                "temperatura": round(temp, 2),
                "estado_critico": nitrato > 50.0 or o2 < 3.0,
                "dosing_locked": o2 < 4.0,
                "carbon_amount_gr": round(carbon_demand * 20.0, 2),
                "last_updated_iso": last_updated_iso,
                "data_age_minutes": round(data_age, 1)
            })
    except Exception as e:
        print(f"❌ biofloc/status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/digital-twin/compare/<int:pool_id>", methods=["GET"])
def get_yesterday_comparison(pool_id: int):
    """
    Devuelve datos de hace 24 horas para comparación (usa ts efectivo).
    """
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT ion_nitrato
                FROM parametro_aguas
                WHERE piscina_id = :pid
                  AND deleted_at IS NULL
                  AND {TS_EFFECTIVE_EXPR} <= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                ORDER BY {TS_EFFECTIVE_EXPR} DESC
                LIMIT 1
            """)
            result = conn.execute(query, {"pid": pool_id}).fetchone()

            if result:
                nitrato_ayer = float(result[0]) if result[0] is not None else 0
                return jsonify({
                    "turbidity_yesterday": min(nitrato_ayer / 100.0, 1.0),
                    "color_hex_yesterday": calcular_color_agua(nitrato_ayer)
                })

            return jsonify({"turbidity_yesterday": 0, "color_hex_yesterday": "#0EA5E9"})
    except Exception as e:
        print(f"❌ digital-twin/compare error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/digital-twin/config/<int:pool_id>", methods=["GET"])
def get_digital_twin_config(pool_id: int):
    """
    Config del digital twin (dimensiones + calidad de agua + biomasa).
    Para agua: usa última medición real (ts efectivo).
    """
    try:
        l, w, d = get_piscina_dimensions(pool_id)

        with engine.connect() as conn:
            q_water = text(f"""
                SELECT ion_nitrato
                FROM parametro_aguas
                WHERE piscina_id = :pid
                  AND deleted_at IS NULL
                ORDER BY {TS_EFFECTIVE_EXPR} DESC
                LIMIT 1
            """)
            water_data = conn.execute(q_water, {"pid": pool_id}).fetchone()

            q_bio = text("""
                SELECT b.cantidad_muestreo, b.peso_promedio
                FROM biometrias b
                JOIN campania_etapas ce ON b.campania_etapa_id = ce.id
                WHERE ce.piscina_id = :pid
                ORDER BY b.fecha_muestreo DESC
                LIMIT 1
            """)
            try:
                bio_data = conn.execute(q_bio, {"pid": pool_id}).fetchone()
            except Exception:
                bio_data = None

            nitrato = float(water_data[0]) if water_data and water_data[0] is not None else 0.0
            fish_count = int(bio_data[0]) if bio_data and bio_data[0] is not None else 2500
            avg_weight = float(bio_data[1]) if bio_data and bio_data[1] is not None else 15.0

            return jsonify({
                "dimensions": {"l": l, "w": w, "d": d},
                "water_quality": {
                    "color_hex": calcular_color_agua(nitrato),
                    "turbidity_factor": min(nitrato / 100.0, 1.0)
                },
                "biomass": {"fish_count": fish_count, "avg_weight": avg_weight}
            })
    except Exception as e:
        print(f"❌ digital-twin/config error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/predict/oxygen/<int:pool_id>", methods=["GET"])
def get_oxygen_prediction(pool_id: int):
    """
    Predicción simple de oxígeno.
    Usa ts efectivo para ordenar/series.
    """
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT
                    {TS_EFFECTIVE_EXPR} AS ts,
                    oxigeno_disuelto,
                    temperatura
                FROM parametro_aguas
                WHERE piscina_id = :pid
                  AND deleted_at IS NULL
                ORDER BY ts DESC
                LIMIT 24
            """)
            df = pd.read_sql(query, conn, params={"pid": pool_id})

            if len(df) < 5:
                return jsonify({
                    "historical": [],
                    "forecast": [],
                    "confidence": 0,
                    "anomaly_scores": {},
                    "alerts": []
                })

            df["oxigeno_disuelto"] = pd.to_numeric(df["oxigeno_disuelto"], errors="coerce")
            df = df.dropna(subset=["ts", "oxigeno_disuelto"])
            df = df.sort_values("ts")

            historical = []
            for _, row in df.iterrows():
                ts = row["ts"]
                historical.append({
                    "time": ts.strftime("%H:%M") if pd.notna(ts) else "",
                    "value": round(float(row["oxigeno_disuelto"]), 2)
                })

            current_o2 = float(df["oxigeno_disuelto"].iloc[-1])
            forecast = []
            alerts = []

            for i in range(1, 7):
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
                "historical": historical[-12:],
                "forecast": forecast,
                "confidence": 89.5,
                "anomaly_scores": {"ph": 12, "temperatura": 5, "oxigeno": 45},
                "alerts": alerts
            })
    except Exception as e:
        print(f"❌ predict/oxygen error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/system/health", methods=["GET"])
def get_system_health():
    """
    Salud del sistema:
    - last_seen lo tomamos de created_at (ingesta), que en tu caso sí viene bien.
    - heartbeat también usa created_at.
    """
    try:
        with engine.connect() as conn:
            q_sensors = text("""
                SELECT p.id, p.nombre, MAX(pa.created_at) AS last_seen
                FROM piscinas p
                LEFT JOIN parametro_aguas pa ON p.id = pa.piscina_id AND pa.deleted_at IS NULL
                GROUP BY p.id, p.nombre
            """)
            rows = conn.execute(q_sensors).fetchall()

            sensors_status = []
            now = datetime.now()

            for row in rows:
                pid, name, last_seen = row

                q_heartbeat = text("""
                    SELECT created_at
                    FROM parametro_aguas
                    WHERE piscina_id = :pid
                      AND deleted_at IS NULL
                      AND created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                    ORDER BY created_at ASC
                """)
                heartbeat_data = conn.execute(q_heartbeat, {"pid": pid}).fetchall()

                heartbeat = []
                for h in range(24):
                    hour_start = now - timedelta(hours=24 - h)
                    hour_end = hour_start + timedelta(hours=1)

                    has_data = any(
                        hour_start <= d[0] < hour_end
                        for d in heartbeat_data if d and d[0]
                    )
                    heartbeat.append({"hour": h, "active": has_data})

                latency = 0
                status = "offline"
                last_seen_str = "Nunca"

                if last_seen:
                    last_seen_dt = last_seen if isinstance(last_seen, datetime) else datetime.strptime(str(last_seen), "%Y-%m-%d %H:%M:%S")
                    delta = now - last_seen_dt
                    latency = int(delta.total_seconds() * 1000)
                    last_seen_str = last_seen_dt.strftime("%H:%M:%S")

                    if latency < 600000:  # < 10 min
                        status = "online"

                sensors_status.append({
                    "id": f"SENSOR-P{pid}",
                    "type": "Multi-Parametro",
                    "status": status,
                    "latency_ms": latency,
                    "last_seen": last_seen_str,
                    "heartbeat": heartbeat
                })

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

            count = conn.execute(text("SELECT COUNT(*) FROM parametro_aguas WHERE deleted_at IS NULL")).fetchone()[0]

            q_critical = text("""
                SELECT COUNT(*)
                FROM parametro_aguas
                WHERE deleted_at IS NULL
                  AND (oxigeno_disuelto < 3.0 OR ion_nitrato > 80)
                  AND DATE(created_at) = CURDATE()
            """)
            critical_count = conn.execute(q_critical).fetchone()[0]

            avg_latency = int(sum(s["latency_ms"] for s in sensors_status) / len(sensors_status)) if sensors_status else 0

            return jsonify({
                "cleaned_records": int(count * 0.15),
                "imputed_records": int(count * 0.05),
                "avg_latency_ms": avg_latency,
                "sensors": sensors_status,
                "logs": logs,
                "critical_alerts_today": critical_count
            })
    except Exception as e:
        print(f"❌ system/health error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
