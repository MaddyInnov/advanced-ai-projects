"""ResolveDesk AI tests — run: python3 tests.py  (uses a temp DB)."""
import os, tempfile

os.environ["RESOLVEDESK_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
import app


def main():
    passed = 0
    def check(name, cond):
        nonlocal passed
        assert cond, f"FAILED: {name}"
        passed += 1; print(f"  ok - {name}")

    app.init_db()
    a, err = app.register_tenant("ShopA", "E-commerce", "a@test.in", "pass1234")
    b, err = app.register_tenant("GymB", "Fitness", "b@test.in", "pass1234")
    check("register two tenants", a and b and a["widget_key"] != b["widget_key"])
    token_a = app.login("a@test.in", "pass1234")
    check("login works / bad login rejected", token_a and app.login("a@test.in", "wrong") is None)

    kid = app.add_kb(a["id"], "Refund policy", "If you are wondering how long a refund takes, the refund amount reaches your bank account within 5-7 business days after the return pickup is completed. Refunds are processed to the original payment method.")
    check("KB add", kid > 0 and len(app.list_kb(a["id"])) == 1)
    doc, score = app.retrieve(a["id"], "how long does a refund take?")
    check("KB retrieve finds refund article", doc and doc["title"] == "Refund policy" and score >= app.CHAT_THRESHOLD)

    res, err = app.chat(a["widget_key"], "How long does a refund take to reach my account?")
    check("chat answers known question", err is None and res["ticket_created"] is False)
    check("chat cites the article", res["citation"] == "Refund policy")

    before = len(app.list_tickets(a["id"]))
    res2, _ = app.chat(a["widget_key"], "Do you offer chartered helicopter delivery to the moon?")
    check("low-confidence question creates a ticket", res2["ticket_created"] is True and len(app.list_tickets(a["id"])) == before + 1)

    from ml import TriageModel
    pred = app.get_model().predict("I want a refund for my damaged product, this is urgent!")
    check("triage: refund intent", pred["intent"] == "refund")
    check("triage: urgent priority", pred["priority"] == "urgent")

    ana = app.analytics(a["id"])
    check("analytics counts match actions", ana["total_conversations"] == 2 and ana["tickets_created"] == before + 1 and ana["ai_resolved"] == 1)

    # tenant isolation
    doc_b, _ = app.retrieve(b["id"], "how long does a refund take?")
    check("tenant isolation: B cannot retrieve A's KB", len(app.list_kb(b["id"])) == 0 and doc_b is None)
    check("tenant isolation: B has no tickets", len(app.list_tickets(b["id"])) == 0 and app.analytics(b["id"])["total_conversations"] == 0)

    tid = app.list_tickets(a["id"])[0]["id"]
    check("ticket status update (owner only)", app.set_ticket_status(a["id"], tid, "resolved") and not app.set_ticket_status(b["id"], tid, "open"))

    print(f"\nALL {passed} TESTS PASSED")


if __name__ == "__main__":
    main()
