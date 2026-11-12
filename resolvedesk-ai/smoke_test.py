"""Full user journey over real HTTP: register -> login -> add KB -> chat ->
ticket -> analytics, plus seeded pages. Starts the server itself.
Run: python3 smoke_test.py
"""
import json, os, subprocess, sys, tempfile, time, urllib.request

PORT = 8199
BASE = f"http://localhost:{PORT}"


def call(method, path, body=None, token=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + token} if token else {})})
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode()
        return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)


def main():
    db = os.path.join(tempfile.mkdtemp(), "smoke.db")
    env = {**os.environ, "PORT": str(PORT), "RESOLVEDESK_DB": db}
    proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    steps = 0
    def check(name, cond):
        nonlocal steps
        assert cond, f"FAILED: {name}"
        steps += 1; print(f"  ok - {name}")
    try:
        for _ in range(50):
            try:
                call("GET", "/"); break
            except Exception: time.sleep(0.3)
        s, html = call("GET", "/"); check("landing page served", s == 200 and "ResolveDesk AI" in html)
        s, html = call("GET", "/dashboard"); check("dashboard served", s == 200 and "Dashboard" in html)
        s, html = call("GET", "/widget-demo"); check("widget demo served", s == 200 and "UrbanKart" in html)

        s, reg = call("POST", "/api/register", {"name": "SmokeShop", "industry": "Retail", "email": "smoke@test.in", "password": "pass1234"})
        check("register tenant", s == 201 and reg["widget_key"].startswith("wk_"))
        s, lg = call("POST", "/api/login", {"email": "smoke@test.in", "password": "pass1234"})
        check("login", s == 200 and lg["token"])
        tok = lg["token"]
        s, kb = call("POST", "/api/kb", {"title": "Store timings", "body": "Our store is open 9 AM to 9 PM every day including Sundays."}, tok)
        check("add KB article", s == 201)
        s, chat = call("POST", "/api/chat", {"widget_key": reg["widget_key"], "message": "What are your store timings?"})
        check("chat answers from KB with citation", s == 200 and chat["citation"] == "Store timings" and not chat["ticket_created"])
        s, chat2 = call("POST", "/api/chat", {"widget_key": reg["widget_key"], "message": "Can I book a helicopter tour?"})
        check("unknown question creates ticket", s == 200 and chat2["ticket_created"])
        s, ana = call("GET", "/api/analytics", token=tok)
        check("analytics reflect journey", ana["total_conversations"] == 2 and ana["tickets_created"] == 1 and ana["ai_resolved"] == 1)
        s, tk = call("GET", "/api/tickets", token=tok)
        check("ticket listed", len(tk["tickets"]) == 1)
        print(f"\nSMOKE TEST PASS — {steps} steps")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
