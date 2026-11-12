"""ResolveDesk AI - multi-tenant AI customer-support SaaS MVP.

Stdlib only: http.server + sqlite3. Demo-grade auth (labelled as such).
Author/owner: Ashutosh Sahoo.
Run:  python3 app.py            (seeds demo tenants, serves on :8090)
"""
import hashlib, json, math, os, re, secrets, sqlite3, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ml import TriageModel, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("RESOLVEDESK_DB", os.path.join(HERE, "resolvedesk.db"))
MODEL_PATH = os.path.join(HERE, "triage_model.npz")
CHAT_THRESHOLD = 0.12  # min TF-IDF cosine to answer from KB instead of ticketing

_model = None


def get_model():
    global _model
    if _model is None:
        _model = TriageModel.load(MODEL_PATH)
        if _model is None:  # train on the fly if model file missing
            from ml import generate_data
            _model = TriageModel().train(generate_data())
    return _model


# ------------------------------------------------------------------- db --
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    c = db()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS tenants(id INTEGER PRIMARY KEY, name TEXT, industry TEXT,
        email TEXT UNIQUE, password_hash TEXT, salt TEXT, widget_key TEXT UNIQUE, created REAL);
    CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, tenant_id INTEGER);
    CREATE TABLE IF NOT EXISTS kb(id INTEGER PRIMARY KEY, tenant_id INTEGER, title TEXT, body TEXT);
    CREATE TABLE IF NOT EXISTS conversations(id INTEGER PRIMARY KEY, tenant_id INTEGER, message TEXT,
        answer TEXT, citation TEXT, confidence REAL, intent TEXT, priority TEXT, sentiment TEXT,
        resolved_by_ai INTEGER, created REAL);
    CREATE TABLE IF NOT EXISTS tickets(id INTEGER PRIMARY KEY, tenant_id INTEGER, conversation_id INTEGER,
        message TEXT, intent TEXT, priority TEXT, sentiment TEXT, status TEXT DEFAULT 'open', created REAL);
    """)
    c.commit(); c.close()


def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 60000).hex()


def register_tenant(name, industry, email, password):
    c = db()
    if c.execute("SELECT 1 FROM tenants WHERE email=?", (email,)).fetchone():
        c.close(); return None, "email already registered"
    salt = secrets.token_hex(8)
    wk = "wk_" + secrets.token_hex(10)
    cur = c.execute("INSERT INTO tenants(name,industry,email,password_hash,salt,widget_key,created) VALUES(?,?,?,?,?,?,?)",
                    (name, industry, email, _hash(password, salt), salt, wk, time.time()))
    c.commit()
    tid = cur.lastrowid; c.close()
    return {"id": tid, "name": name, "widget_key": wk}, None


def login(email, password):
    c = db()
    row = c.execute("SELECT * FROM tenants WHERE email=?", (email,)).fetchone()
    if not row or _hash(password, row["salt"]) != row["password_hash"]:
        c.close(); return None
    token = secrets.token_hex(16)
    c.execute("INSERT INTO sessions VALUES(?,?)", (token, row["id"])); c.commit(); c.close()
    return token


def tenant_for_token(token):
    if not token: return None
    c = db()
    row = c.execute("SELECT t.* FROM sessions s JOIN tenants t ON t.id=s.tenant_id WHERE s.token=?", (token,)).fetchone()
    c.close(); return dict(row) if row else None


def tenant_for_widget_key(key):
    c = db()
    row = c.execute("SELECT * FROM tenants WHERE widget_key=?", (key,)).fetchone()
    c.close(); return dict(row) if row else None


# ------------------------------------------------------------------- kb --
def add_kb(tenant_id, title, body):
    c = db()
    cur = c.execute("INSERT INTO kb(tenant_id,title,body) VALUES(?,?,?)", (tenant_id, title, body))
    c.commit(); kid = cur.lastrowid; c.close(); return kid


def list_kb(tenant_id):
    c = db()
    rows = [dict(r) for r in c.execute("SELECT * FROM kb WHERE tenant_id=? ORDER BY id", (tenant_id,))]
    c.close(); return rows


_STOP = set("a an the is are do does did can i you your yours we our it to of for in on at and or how what when will would should me my this that with after".split())


def _rtok(text):
    """Retrieval tokens: stopwords removed + light plural stemming."""
    out = []
    for t in tokenize(text):
        if t in _STOP: continue
        if len(t) > 3 and t.endswith("s"): t = t[:-1]
        out.append(t)
    return out


def retrieve(tenant_id, query):
    """TF-IDF cosine retrieval over THIS tenant's KB only. Returns (doc, score)."""
    docs = list_kb(tenant_id)
    if not docs: return None, 0.0
    corpus = [d["title"] + " " + d["title"] + " " + d["body"] for d in docs]
    N = len(corpus)
    df = {}
    for d in corpus:
        for t in set(_rtok(d)): df[t] = df.get(t, 0) + 1

    def vec(text):
        counts = {}
        for t in _rtok(text): counts[t] = counts.get(t, 0) + 1
        return {t: (1 + math.log(cnt)) * (math.log((1 + N) / (1 + df.get(t, 0))) + 1)
                for t, cnt in counts.items()}

    qv = vec(query); qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
    best, best_score = None, 0.0
    for doc, text in zip(docs, corpus):
        dv = vec(text); dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
        score = sum(qv[t] * dv.get(t, 0.0) for t in qv) / (qn * dn)
        if score > best_score: best, best_score = doc, score
    return best, float(best_score)


# ----------------------------------------------------------------- chat --
def chat(widget_key, message):
    tenant = tenant_for_widget_key(widget_key)
    if not tenant: return None, "invalid widget key"
    triage = get_model().predict(message)
    doc, score = retrieve(tenant["id"], message)
    c = db()
    if doc and score >= CHAT_THRESHOLD:
        answer = doc["body"]; citation = doc["title"]; resolved = 1; ticket_id = None
    else:
        answer = ("Thanks for reaching out. I couldn't find a confident answer in our help "
                  "articles, so I've created a support ticket — our team will get back to you.")
        citation = None; resolved = 0
        cur = c.execute("INSERT INTO conversations(tenant_id,message,answer,citation,confidence,intent,priority,sentiment,resolved_by_ai,created) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (tenant["id"], message, answer, None, score, triage["intent"], triage["priority"], triage["sentiment"], 0, time.time()))
        conv_id = cur.lastrowid
        cur2 = c.execute("INSERT INTO tickets(tenant_id,conversation_id,message,intent,priority,sentiment,status,created) VALUES(?,?,?,?,?,?,?,?)",
                         (tenant["id"], conv_id, message, triage["intent"], triage["priority"], triage["sentiment"], "open", time.time()))
        ticket_id = cur2.lastrowid; c.commit(); c.close()
        return {"answer": answer, "citation": None, "confidence": round(score, 4),
                "ticket_created": True, "ticket_id": ticket_id, **triage}, None
    c.execute("INSERT INTO conversations(tenant_id,message,answer,citation,confidence,intent,priority,sentiment,resolved_by_ai,created) VALUES(?,?,?,?,?,?,?,?,?,?)",
              (tenant["id"], message, answer, citation, score, triage["intent"], triage["priority"], triage["sentiment"], 1, time.time()))
    c.commit(); c.close()
    return {"answer": answer, "citation": citation, "confidence": round(score, 4),
            "ticket_created": False, "ticket_id": None, **triage}, None


def list_tickets(tenant_id):
    c = db()
    order = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
    rows = [dict(r) for r in c.execute("SELECT * FROM tickets WHERE tenant_id=?", (tenant_id,))]
    c.close()
    rows.sort(key=lambda r: (order.get(r["priority"], 2), -r["created"]))
    return rows


def set_ticket_status(tenant_id, ticket_id, status):
    if status not in ("open", "in-progress", "resolved"): return False
    c = db()
    cur = c.execute("UPDATE tickets SET status=? WHERE id=? AND tenant_id=?", (status, ticket_id, tenant_id))
    c.commit(); ok = cur.rowcount > 0; c.close(); return ok


def analytics(tenant_id):
    c = db()
    convs = c.execute("SELECT COUNT(*) n, COALESCE(SUM(resolved_by_ai),0) r, COALESCE(AVG(confidence),0) a FROM conversations WHERE tenant_id=?", (tenant_id,)).fetchone()
    tickets = c.execute("SELECT COUNT(*) n FROM tickets WHERE tenant_id=?", (tenant_id,)).fetchone()["n"]
    intents = {r["intent"]: r["n"] for r in c.execute("SELECT intent, COUNT(*) n FROM conversations WHERE tenant_id=? GROUP BY intent", (tenant_id,))}
    c.close()
    total = convs["n"]
    return {"total_conversations": total, "ai_resolved": int(convs["r"]),
            "deflection_rate": round(convs["r"] / total, 4) if total else 0.0,
            "tickets_created": tickets, "avg_confidence": round(float(convs["a"]), 4),
            "intent_breakdown": intents}


# ----------------------------------------------------------------- seed --
URBANKART_KB = [
    ("How to track my order", "You can track your order from the Orders page in your UrbanKart account. Enter your order ID to see live delivery status, the courier partner and the expected delivery date. Tracking updates every few hours."),
    ("Delivery charges and timelines", "Standard delivery is free on orders above Rs 499 and Rs 49 below that. Delivery takes 2-4 business days in metro cities and 4-7 business days in other areas. Express next-day delivery is available in select pincodes for Rs 99."),
    ("Returns and refunds policy", "You can return most products within 7 days of delivery if they are unused and in original packaging. Refunds are processed to the original payment method within 5-7 business days after the returned item passes quality check. Refund status is visible on the Orders page."),
    ("How to cancel an order", "Orders can be cancelled free of charge from the Orders page before they are shipped. Once shipped, you can refuse delivery or raise a return after delivery. Prepaid amounts are refunded within 5-7 business days."),
    ("Payment methods accepted", "UrbanKart accepts UPI, credit and debit cards, net banking, wallets and Cash on Delivery. EMI is available on cards for orders above Rs 3000. All payments are processed over encrypted gateways."),
    ("Warranty on electronics", "Electronics sold on UrbanKart carry the brand warranty, typically 1 year from delivery. Keep your invoice for warranty claims at any authorised brand service centre. Extended warranty plans can be added at checkout."),
    ("How to change my delivery address", "You can change the delivery address from the Address Book in your account before the order is shipped. Address cannot be changed after dispatch; in that case contact support and we will try to reroute the courier."),
    ("Damaged or wrong product received", "If you receive a damaged or wrong product, raise a return within 48 hours with photos from the Orders page. We arrange a free pickup and ship a replacement or refund you in full after the pickup."),
    ("Offers, coupons and loyalty points", "Apply coupon codes at checkout in the promo box. UrbanKart loyalty points: you earn 1 point per Rs 100 spent and 100 points equal Rs 25 off. Points expire after 12 months."),
    ("Account, password and login help", "Use Forgot Password on the login page to reset your password by email OTP. If your account is locked after repeated attempts, it unlocks automatically in 30 minutes. Enable two-factor authentication from Settings for extra security."),
]
FITLIFE_KB = [
    ("Membership plans and pricing", "FitLife Gym plans: Monthly Rs 1500, Quarterly Rs 3900, Annual Rs 12000. All plans include cardio, weights and one group class per day. Personal training is Rs 800 per session or Rs 8000 for 12 sessions."),
    ("How to freeze my membership", "Annual and quarterly members can freeze their membership once for up to 30 days for medical or travel reasons. Raise a freeze request at the front desk or from the app at least 3 days in advance."),
    ("Class schedule and booking", "Group classes run at 6 AM, 7 AM, 6 PM and 7 PM. Book a slot in the FitLife app up to 24 hours ahead. Zumba is on Monday and Thursday, Yoga on Tuesday and Friday, CrossFit on Wednesday and Saturday."),
    ("Refund and cancellation policy", "Membership fees are refundable within the first 7 days if no sessions were used. After that, fees are non-refundable but you may transfer your membership to another person once for Rs 500."),
    ("Timings and holidays", "FitLife is open 5 AM to 10 PM on weekdays, 6 AM to 8 PM on weekends. The gym is closed on major public holidays; the holiday list is posted on the notice board and app."),
]


def seed():
    c = db()
    existing = c.execute("SELECT COUNT(*) n FROM tenants").fetchone()["n"]; c.close()
    if existing: return
    uk, _ = register_tenant("UrbanKart", "E-commerce", "demo@urbankart.in", "demo1234")
    for t, b in URBANKART_KB: add_kb(uk["id"], t, b)
    fl, _ = register_tenant("FitLife Gym", "Fitness", "demo@fitlife.in", "demo1234")
    for t, b in FITLIFE_KB: add_kb(fl["id"], t, b)
    print(f"Seeded UrbanKart (widget_key={uk['widget_key']}) and FitLife Gym (widget_key={fl['widget_key']})")


# ----------------------------------------------------------------- http --
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        if ctype != "application/json": data = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _auth(self):
        return tenant_for_token(self.headers.get("Authorization", "").replace("Bearer ", ""))

    def _static(self, name):
        path = os.path.join(HERE, "web", name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f: self._send(200, f.read(), "text/html; charset=utf-8")
        else: self._send(404, {"error": "not found"})

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/": return self._static("index.html")
        if p == "/dashboard": return self._static("dashboard.html")
        if p == "/widget-demo": return self._static("widget.html")
        if p == "/api/kb":
            t = self._auth()
            if not t: return self._send(401, {"error": "unauthorized"})
            return self._send(200, {"kb": list_kb(t["id"])})
        if p == "/api/tickets":
            t = self._auth()
            if not t: return self._send(401, {"error": "unauthorized"})
            return self._send(200, {"tickets": list_tickets(t["id"])})
        if p == "/api/analytics":
            t = self._auth()
            if not t: return self._send(401, {"error": "unauthorized"})
            return self._send(200, analytics(t["id"]))
        if p == "/api/me":
            t = self._auth()
            if not t: return self._send(401, {"error": "unauthorized"})
            return self._send(200, {"id": t["id"], "name": t["name"], "industry": t["industry"], "widget_key": t["widget_key"]})
        if p == "/api/demo-tenants":
            c = db(); rows = [dict(r) for r in c.execute("SELECT name, industry, widget_key FROM tenants")]; c.close()
            return self._send(200, {"tenants": rows})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        p = urlparse(self.path).path
        try: body = self._body()
        except Exception: return self._send(400, {"error": "bad json"})
        if p == "/api/register":
            t, err = register_tenant(body.get("name", ""), body.get("industry", ""), body.get("email", ""), body.get("password", ""))
            return self._send(400, {"error": err}) if err else self._send(201, t)
        if p == "/api/login":
            token = login(body.get("email", ""), body.get("password", ""))
            return self._send(200, {"token": token}) if token else self._send(401, {"error": "invalid credentials"})
        if p == "/api/chat":
            res, err = chat(body.get("widget_key", ""), body.get("message", ""))
            return self._send(404, {"error": err}) if err else self._send(200, res)
        if p == "/api/kb":
            t = self._auth()
            if not t: return self._send(401, {"error": "unauthorized"})
            kid = add_kb(t["id"], body.get("title", ""), body.get("body", ""))
            return self._send(201, {"id": kid})
        m = re.match(r"/api/tickets/(\d+)/status", p)
        if m:
            t = self._auth()
            if not t: return self._send(401, {"error": "unauthorized"})
            ok = set_ticket_status(t["id"], int(m.group(1)), body.get("status", ""))
            return self._send(200, {"ok": ok}) if ok else self._send(404, {"error": "ticket not found"})
        return self._send(404, {"error": "not found"})

    def do_DELETE(self):
        p = urlparse(self.path).path
        t = self._auth()
        if not t: return self._send(401, {"error": "unauthorized"})
        m = re.match(r"/api/kb/(\d+)", p)
        if m:
            c = db(); c.execute("DELETE FROM kb WHERE id=? AND tenant_id=?", (int(m.group(1)), t["id"]))
            c.commit(); c.close(); return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


def main():
    init_db(); seed()
    port = int(os.environ.get("PORT", "8090"))
    print(f"ResolveDesk AI running at http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
