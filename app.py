"""
Concrete curing monitor: ESP32 temperature ingest, live graph, maturity (Nurse-Saul,
ASTM C1074) and strength estimation from a user-supplied calibration curve.

Deployable on Render: uses PostgreSQL when DATABASE_URL is set (data survives redeploys),
otherwise falls back to a local SQLite file for development.

Local run:  pip install -r requirements.txt && API_KEY=mysecret python app.py
Render:     gunicorn app:app   (see render.yaml)
"""
import os
import sqlite3
import time

import numpy as np
from flask import Flask, g, jsonify, render_template_string, request

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):          # older URL scheme some hosts still hand out
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]
USE_PG = DATABASE_URL.startswith("postgresql://")
if USE_PG:
    import psycopg2
    import psycopg2.extras

DB_PATH = os.environ.get("DB_PATH", "curing.db")    # SQLite fallback only
API_KEY = os.environ.get("API_KEY", "change-me")
DATUM_TEMP = float(os.environ.get("DATUM_TEMP", 0.0))   # Nurse-Saul datum temperature (deg C)
REF_TEMP = float(os.environ.get("REF_TEMP", 27.0))      # standard curing temp for cube tests (IS 456)
MAX_GAP_S = 2 * 3600                                    # cap on a single integration step (sensor outage)

app = Flask(__name__)


# ----------------------------------------------------------------- database
class DB:
    """Thin wrapper so the same SQL (with ? placeholders) runs on SQLite and PostgreSQL."""

    def __init__(self):
        if USE_PG:
            self.conn = psycopg2.connect(DATABASE_URL)
        else:
            self.conn = sqlite3.connect(DB_PATH)
            self.conn.row_factory = sqlite3.Row

    def execute(self, sql, args=()):
        if USE_PG:
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql.replace("?", "%s"), args)
            return cur
        return self.conn.execute(sql, args)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


def get_db():
    if "db" not in g:
        g.db = DB()
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    pk = "SERIAL PRIMARY KEY" if USE_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
    real = "DOUBLE PRECISION" if USE_PG else "REAL"   # PG "REAL" is float4: too coarse for timestamps
    db = DB()
    for stmt in (
        f"CREATE TABLE IF NOT EXISTS readings(id {pk}, sensor_id TEXT NOT NULL, ts {real} NOT NULL, temp {real} NOT NULL)",
        "CREATE INDEX IF NOT EXISTS idx_readings ON readings(sensor_id, ts)",
        f"CREATE TABLE IF NOT EXISTS calibration(id {pk}, age_days {real} NOT NULL, strength {real} NOT NULL, maturity {real})",
        "CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT)",
    ):
        db.execute(stmt)
    db.commit()
    db.close()


# ------------------------------------------------------------------- maths
def compute_maturity(rows):
    """Nurse-Saul maturity M = sum((Ta - T0) * dt) in deg C-hours (trapezoid rule)."""
    m, prev = 0.0, None
    for ts, t in rows:
        if prev is not None:
            dt = min(ts - prev[0], MAX_GAP_S) / 3600.0
            m += max((t + prev[1]) / 2.0 - DATUM_TEMP, 0.0) * dt
        prev = (ts, t)
    return m


def standard_maturity(age_days):
    """Maturity of a cube cured at the reference temperature for age_days."""
    return (REF_TEMP - DATUM_TEMP) * 24.0 * age_days


def fit_curve():
    """Plowman log fit: S = a + b*ln(M). Returns dict or None."""
    rows = get_db().execute("SELECT age_days, strength, maturity FROM calibration").fetchall()
    pts = []
    for r in rows:
        m = r["maturity"] if r["maturity"] else standard_maturity(r["age_days"])
        if m > 0:
            pts.append((m, r["strength"]))
    if len({round(p[0], 3) for p in pts}) < 2:
        return None
    x = np.log([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"a": float(a), "b": float(b), "r2": r2, "n": len(pts)}


def strength_at(fit, maturity):
    if not fit or maturity <= 0:
        return None
    return max(fit["a"] + fit["b"] * float(np.log(maturity)), 0.0)


def get_pour_start(sensor_id=None):
    row = get_db().execute("SELECT value FROM settings WHERE key='pour_start'").fetchone()
    if row:
        return float(row["value"])
    q = "SELECT MIN(ts) AS t FROM readings"
    args = ()
    if sensor_id:
        q += " WHERE sensor_id=?"
        args = (sensor_id,)
    return get_db().execute(q, args).fetchone()["t"]


# --------------------------------------------------------------------- API
@app.get("/healthz")
def healthz():
    return "ok"


@app.post("/api/data")
def ingest():
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="unauthorized"), 401
    d = request.get_json(silent=True) or {}
    try:
        temp = float(d["temperature"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="'temperature' (number) required"), 400
    if not -40 <= temp <= 150:
        return jsonify(error="temperature out of range"), 400
    sensor = str(d.get("sensor_id", "slab1"))[:40]
    ts = float(d["timestamp"]) if d.get("timestamp") else time.time()
    db = get_db()
    db.execute("INSERT INTO readings(sensor_id, ts, temp) VALUES(?,?,?)", (sensor, ts, temp))
    db.commit()
    return jsonify(ok=True)


@app.get("/api/sensors")
def sensors():
    rows = get_db().execute("SELECT DISTINCT sensor_id FROM readings ORDER BY sensor_id").fetchall()
    return jsonify([r["sensor_id"] for r in rows])


@app.get("/api/readings")
def readings():
    sensor = request.args.get("sensor", "slab1")
    limit = min(int(request.args.get("limit", 300)), 5000)
    rows = get_db().execute(
        "SELECT ts, temp FROM readings WHERE sensor_id=? ORDER BY ts DESC LIMIT ?", (sensor, limit)
    ).fetchall()
    rows.reverse()
    return jsonify([{"ts": r["ts"], "temp": r["temp"]} for r in rows])


@app.get("/api/status")
def status():
    sensor = request.args.get("sensor", "slab1")
    db = get_db()
    last = db.execute(
        "SELECT ts, temp FROM readings WHERE sensor_id=? ORDER BY ts DESC LIMIT 1", (sensor,)
    ).fetchone()
    if not last:
        return jsonify(has_data=False)
    start = get_pour_start(sensor)
    rows = db.execute(
        "SELECT ts, temp FROM readings WHERE sensor_id=? AND ts>=? ORDER BY ts", (sensor, start)
    ).fetchall()
    m = compute_maturity([(r["ts"], r["temp"]) for r in rows])
    fit = fit_curve()
    s = strength_at(fit, m)
    s28 = strength_at(fit, standard_maturity(28))
    return jsonify(
        has_data=True,
        temp=last["temp"],
        last_ts=last["ts"],
        pour_start=start,
        hours_since_pour=(last["ts"] - start) / 3600.0,
        maturity=m,
        equivalent_age_days=m / ((REF_TEMP - DATUM_TEMP) * 24.0),
        calibrated=fit is not None,
        strength=s,
        pct_of_28d=(100.0 * s / s28) if (s is not None and s28) else None,
        datum=DATUM_TEMP,
        ref_temp=REF_TEMP,
    )


@app.post("/api/pour")
def set_pour():
    """Mark the pour time (defaults to now). Maturity counts from here."""
    d = request.get_json(silent=True) or {}
    ts = float(d["timestamp"]) if d.get("timestamp") else time.time()
    db = get_db()
    db.execute(
        "INSERT INTO settings(key, value) VALUES('pour_start', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(ts),),
    )
    db.commit()
    return jsonify(ok=True, pour_start=ts)


@app.get("/api/calibration")
def calibration_get():
    db = get_db()
    rows = db.execute("SELECT * FROM calibration ORDER BY age_days").fetchall()
    pts = []
    for r in rows:
        m = r["maturity"] if r["maturity"] else standard_maturity(r["age_days"])
        pts.append({"id": r["id"], "age_days": r["age_days"], "strength": r["strength"],
                    "maturity": m, "maturity_given": bool(r["maturity"])})
    fit = fit_curve()
    curve = []
    if fit and pts:
        lo = max(min(p["maturity"] for p in pts) * 0.5, 1.0)
        hi = max(p["maturity"] for p in pts) * 1.3
        for m in np.geomspace(lo, hi, 60):
            curve.append({"x": float(m), "y": strength_at(fit, float(m))})
    return jsonify(points=pts, fit=fit, curve=curve)


@app.post("/api/calibration")
def calibration_add():
    d = request.get_json(silent=True) or {}
    try:
        age = float(d["age_days"])
        strength = float(d["strength"])
        maturity = float(d["maturity"]) if d.get("maturity") not in (None, "") else None
    except (KeyError, TypeError, ValueError):
        return jsonify(error="age_days and strength (numbers) required"), 400
    if age <= 0 or strength < 0:
        return jsonify(error="age must be > 0 and strength >= 0"), 400
    db = get_db()
    db.execute("INSERT INTO calibration(age_days, strength, maturity) VALUES(?,?,?)",
               (age, strength, maturity))
    db.commit()
    return jsonify(ok=True)


@app.delete("/api/calibration/<int:cid>")
def calibration_delete(cid):
    db = get_db()
    db.execute("DELETE FROM calibration WHERE id=?", (cid,))
    db.commit()
    return jsonify(ok=True)


# --------------------------------------------------------------- dashboard
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Concrete Curing Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f4f5f7;color:#222}
 header{background:#243447;color:#fff;padding:12px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 header h1{font-size:18px;margin:0;flex:1}
 main{max-width:1100px;margin:16px auto;padding:0 12px}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:14px}
 .card,.panel{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px #0002}
 .card small{color:#666;display:block}.card b{font-size:24px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 @media(max-width:800px){.grid{grid-template-columns:1fr}}
 input,button,select{padding:6px 8px;font-size:14px}
 button{cursor:pointer}
 table{border-collapse:collapse;width:100%;font-size:14px;margin-top:8px}
 td,th{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left}
 .muted{color:#777;font-size:13px}
 .bar{height:10px;background:#e3e6ea;border-radius:5px;overflow:hidden;margin-top:6px}
 .bar i{display:block;height:100%;background:#2e8b57;width:0}
</style></head><body>
<header>
 <h1>Concrete Curing Monitor</h1>
 <label>Sensor <select id="sensor"></select></label>
 <button onclick="markPour()">Mark pour = now</button>
</header>
<main>
 <div class="cards">
  <div class="card"><small>Temperature</small><b id="temp">--</b> &deg;C<small id="last"></small></div>
  <div class="card"><small>Maturity (Nurse-Saul)</small><b id="mat">--</b> &deg;C&middot;h</div>
  <div class="card"><small>Equivalent age @ <span id="ref"></span>&deg;C</small><b id="eqage">--</b> days<small id="since"></small></div>
  <div class="card"><small>Estimated strength</small><b id="str">--</b> MPa
   <div class="bar"><i id="pctbar"></i></div><small id="pct"></small></div>
 </div>
 <div class="grid">
  <div class="panel"><h3>Live temperature</h3><canvas id="tchart" height="220"></canvas></div>
  <div class="panel"><h3>Strength&ndash;maturity calibration</h3><canvas id="cchart" height="220"></canvas>
   <div class="muted" id="fit"></div></div>
 </div>
 <div class="panel" style="margin-top:14px">
  <h3>Calibration data (cube tests)</h3>
  <form onsubmit="addCal(event)">
   Age (days) <input id="age" type="number" step="any" min="0.01" required style="width:80px">
   Strength (MPa) <input id="cs" type="number" step="any" min="0" required style="width:90px">
   Maturity (&deg;C&middot;h, optional) <input id="cm" type="number" step="any" style="width:110px">
   <button>Add</button>
  </form>
  <p class="muted">Enter results at 3, 7, 21, 28 days etc. If maturity is left blank, standard curing at the
   reference temperature is assumed (maturity = (ref &minus; datum) &times; 24 &times; age). If the cubes were
   cured with their own logger, enter their measured maturity instead. At least 2 points are needed; 4+ recommended.</p>
  <table><thead><tr><th>Age (d)</th><th>Strength (MPa)</th><th>Maturity (&deg;C&middot;h)</th><th></th></tr></thead>
  <tbody id="caltable"></tbody></table>
 </div>
</main>
<script>
const $ = id => document.getElementById(id);
const fmt = (v, d=1) => v == null ? '--' : Number(v).toFixed(d);
const sensorSel = $('sensor');
const tchart = new Chart($('tchart'), {type:'line',
  data:{labels:[],datasets:[{label:'Temp (°C)',data:[],borderColor:'#d9480f',tension:.25,pointRadius:0}]},
  options:{animation:false,scales:{x:{ticks:{maxTicksLimit:6}}}}});
const cchart = new Chart($('cchart'), {type:'scatter',
  data:{datasets:[
   {label:'Cube tests',data:[],backgroundColor:'#1c7ed6',pointRadius:5},
   {label:'Fitted curve',data:[],type:'line',borderColor:'#2e8b57',pointRadius:0,showLine:true},
   {label:'Slab now',data:[],backgroundColor:'#d9480f',pointStyle:'triangle',pointRadius:8}]},
  options:{animation:false,scales:{x:{type:'logarithmic',title:{display:true,text:'Maturity (°C·h)'}},
   y:{title:{display:true,text:'Strength (MPa)'},beginAtZero:true}}}});

async function j(url, opts){ const r = await fetch(url, opts); return r.json(); }

async function loadSensors(){
  const list = await j('/api/sensors');
  const cur = sensorSel.value;
  sensorSel.innerHTML = '';
  (list.length ? list : ['slab1']).forEach(s => sensorSel.add(new Option(s, s)));
  if (list.includes(cur)) sensorSel.value = cur;
}

async function refresh(){
  const s = sensorSel.value || 'slab1';
  const [rd, st] = await Promise.all([j('/api/readings?sensor='+s+'&limit=300'), j('/api/status?sensor='+s)]);
  tchart.data.labels = rd.map(r => new Date(r.ts*1000).toLocaleTimeString());
  tchart.data.datasets[0].data = rd.map(r => r.temp);
  tchart.update();
  if (!st.has_data) return;
  $('temp').textContent = fmt(st.temp);
  $('last').textContent = new Date(st.last_ts*1000).toLocaleString();
  $('mat').textContent = fmt(st.maturity, 0);
  $('ref').textContent = st.ref_temp;
  $('eqage').textContent = fmt(st.equivalent_age_days, 2);
  $('since').textContent = fmt(st.hours_since_pour, 1) + ' h since pour';
  $('str').textContent = st.calibrated ? fmt(st.strength) : 'n/a';
  $('pct').textContent = st.calibrated ? fmt(st.pct_of_28d,0)+'% of estimated 28-day strength' : 'Add calibration data';
  $('pctbar').style.width = st.calibrated ? Math.min(st.pct_of_28d,100)+'%' : '0';
  cchart.data.datasets[2].data = (st.calibrated && st.maturity > 0) ? [{x:st.maturity,y:st.strength}] : [];
  cchart.update();
}

async function loadCal(){
  const c = await j('/api/calibration');
  const tb = $('caltable'); tb.innerHTML = '';
  c.points.forEach(p => {
    const tr = tb.insertRow();
    tr.insertCell().textContent = p.age_days;
    tr.insertCell().textContent = p.strength;
    tr.insertCell().textContent = fmt(p.maturity,0) + (p.maturity_given ? '' : ' (std. curing)');
    const b = document.createElement('button'); b.textContent = 'Delete';
    b.onclick = async () => { await fetch('/api/calibration/'+p.id,{method:'DELETE'}); loadCal(); refresh(); };
    tr.insertCell().appendChild(b);
  });
  cchart.data.datasets[0].data = c.points.map(p => ({x:p.maturity,y:p.strength}));
  cchart.data.datasets[1].data = c.curve;
  cchart.update();
  $('fit').textContent = c.fit
    ? `S = ${fmt(c.fit.a,2)} + ${fmt(c.fit.b,2)} · ln(M)   |   R² = ${fmt(c.fit.r2,3)}   |   n = ${c.fit.n}`
    : 'Need at least 2 calibration points with different maturity.';
}

async function addCal(e){
  e.preventDefault();
  const body = {age_days:$('age').value, strength:$('cs').value, maturity:$('cm').value};
  const r = await j('/api/calibration',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if (r.error) return alert(r.error);
  e.target.reset(); loadCal(); refresh();
}

async function markPour(){
  if (!confirm('Reset maturity clock to now?')) return;
  await j('/api/pour',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  refresh();
}

sensorSel.onchange = refresh;
(async()=>{ await loadSensors(); await loadCal(); await refresh();
  setInterval(refresh, 5000); setInterval(loadSensors, 30000); })();
</script></body></html>"""




@app.get("/")
def index():
    return render_template_string(PAGE)


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
