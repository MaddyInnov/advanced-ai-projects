"""ResolveDesk AI - ML triage: intent + priority classifier and sentiment.

TF-IDF features + multinomial logistic regression implemented in NumPy
(no sklearn dependency). Two heads: intent (6 classes) and priority (4).
Author: Ashutosh Sahoo.
"""
import json, math, random, re
import numpy as np

INTENTS = ["billing", "technical", "delivery", "refund", "account", "general"]
PRIORITIES = ["low", "normal", "high", "urgent"]

# ---------------------------------------------------------------- data ---
_TEMPLATES = {
    "billing": [
        "I was charged {amt} twice for my subscription this month",
        "Why is my invoice higher than last month? I see an extra charge of {amt}",
        "My payment failed but the money was deducted from my card",
        "I need a copy of my GST invoice for order {oid}",
        "How do I update my billing address and payment method?",
        "There is an unknown charge of {amt} on my statement from your store",
        "My card was charged even though I cancelled the plan",
        "Can you explain the charges on my latest bill?",
    ],
    "technical": [
        "The app crashes every time I try to open the checkout page",
        "I keep getting error 500 when I upload my profile photo",
        "Your website is not loading on my phone since yesterday",
        "The login page shows a blank screen after I enter the OTP",
        "Notifications stopped working after the latest update",
        "The app freezes when I search for products",
        "I cannot download my report, the button does nothing",
        "The page keeps timing out when I try to pay",
    ],
    "delivery": [
        "Where is my order {oid}? It was supposed to arrive yesterday",
        "My package shows delivered but I never received it",
        "Delivery is delayed by three days, what is happening with order {oid}?",
        "The delivery agent did not call me and marked the attempt as failed",
        "Can I change the delivery address for order {oid}?",
        "My order {oid} is stuck at the hub for two days",
        "Half of my order arrived, the rest is missing",
        "When will my order {oid} be shipped?",
    ],
    "refund": [
        "I want a refund for order {oid}, the product is damaged",
        "Please refund {amt} to my bank account, I returned the item last week",
        "My refund has not arrived even after 10 days",
        "I cancelled my order but the refund is still pending",
        "The shoes I received are the wrong size, I want my money back",
        "Refund the amount for the defective phone I returned",
        "How long does a refund take? I am still waiting for {amt}",
        "I was promised a refund but only received store credit",
    ],
    "account": [
        "I forgot my password and cannot reset it, the link expired",
        "How do I change the email address on my account?",
        "My account is locked after too many login attempts",
        "I want to delete my account and all my data",
        "Two accounts were created with my number, please merge them",
        "I cannot verify my mobile number, OTP never arrives for my account",
        "How do I enable two factor authentication on my account?",
        "My membership shows inactive even though I paid",
    ],
    "general": [
        "What are your customer support working hours?",
        "Do you have a store in Hyderabad?",
        "How do I contact a human agent?",
        "What payment methods do you accept?",
        "Do you offer bulk or corporate orders?",
        "Where can I read your privacy policy?",
        "Do you ship to rural areas?",
        "What is the warranty period for electronics?",
    ],
}
_PRIORITY_PHRASES = {
    "low": ["No hurry, whenever you get time.", "Just wondering, no rush at all.",
            "Answer at your convenience please.", "This is a low priority question."],
    "normal": ["Please let me know.", "Kindly help with this.", "Awaiting your response.", ""],
    "high": ["Please resolve this today, it is important.", "I need this fixed soon, please escalate.",
             "This is affecting my work, please help quickly.", "Need a resolution by end of day."],
    "urgent": ["URGENT: this is an emergency, I need help immediately!",
               "This is extremely urgent, resolve it right now or I will dispute the charge!",
               "Critical issue, I am completely blocked and losing money every hour!",
               "ASAP! Very urgent, my account is compromised and I need instant help!"],
}

POS_WORDS = set("good great excellent happy satisfied thanks thank awesome perfect love helpful fast easy amazing".split())
NEG_WORDS = set("bad terrible worst angry frustrated disappointed refund broken fail failed error delayed missing wrong poor horrible useless furious cheating fraud".split())


def _fill(tpl, rng):
    return tpl.format(amt=rng.choice(["₹499", "₹1,299", "₹2,999", "₹750"]),
                      oid=rng.choice(["#UK10231", "#UK88410", "#OD5521", "#IN90210"]))


def generate_data(n_per_combo=25, seed=42):
    """~600 synthetic-but-realistic support messages, labelled intent+priority."""
    rng = random.Random(seed)
    data = []
    for intent, templates in _TEMPLATES.items():
        for priority, phrases in _PRIORITY_PHRASES.items():
            for _ in range(n_per_combo):
                msg = _fill(rng.choice(templates), rng)
                ph = rng.choice(phrases)
                text = (msg + " " + ph).strip()
                data.append({"text": text, "intent": intent, "priority": priority})
    rng.shuffle(data)
    return data


# ---------------------------------------------------------------- tfidf --
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Tfidf:
    def __init__(self, max_vocab=3000):
        self.vocab, self.idf = {}, None
        self.max_vocab = max_vocab

    def fit(self, docs):
        df = {}
        for d in docs:
            for t in set(tokenize(d)):
                df[t] = df.get(t, 0) + 1
        terms = sorted(df.items(), key=lambda kv: -kv[1])[: self.max_vocab]
        self.vocab = {t: i for i, (t, _) in enumerate(terms)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t, _ in terms])
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for r, d in enumerate(docs):
            counts = {}
            for t in tokenize(d):
                if t in self.vocab:
                    counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                X[r, self.vocab[t]] = (1 + math.log(c)) * self.idf[self.vocab[t]]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1, norms)


# ------------------------------------------------------------- classifier -
class SoftmaxLR:
    def __init__(self, n_features, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_features, n_classes))
        self.b = np.zeros(n_classes)

    def fit(self, X, y, epochs=300, lr=0.5, l2=1e-4):
        n = len(y)
        Y = np.eye(self.W.shape[1])[y]
        for _ in range(epochs):
            logits = X @ self.W + self.b
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
            G = (P - Y) / n
            self.W -= lr * (X.T @ G + l2 * self.W)
            self.b -= lr * G.sum(axis=0)
        return self

    def predict_proba(self, X):
        logits = X @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits); return P / P.sum(axis=1, keepdims=True)


def sentiment(text):
    toks = tokenize(text)
    pos = sum(t in POS_WORDS for t in toks); neg = sum(t in NEG_WORDS for t in toks)
    if neg > pos: return "negative"
    if pos > neg: return "positive"
    return "neutral"


def macro_f1(y_true, y_pred, labels):
    f1s = []
    for lab in labels:
        tp = int(((y_pred == lab) & (y_true == lab)).sum())
        fp = int(((y_pred == lab) & (y_true != lab)).sum())
        fn = int(((y_pred != lab) & (y_true == lab)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


class TriageModel:
    def __init__(self):
        self.tfidf = Tfidf()
        self.intent_clf = None; self.priority_clf = None

    def train(self, data):
        texts = [d["text"] for d in data]
        X = self.tfidf.fit(texts).transform(texts)
        yi = np.array([INTENTS.index(d["intent"]) for d in data])
        yp = np.array([PRIORITIES.index(d["priority"]) for d in data])
        self.intent_clf = SoftmaxLR(X.shape[1], len(INTENTS), seed=1).fit(X, yi)
        self.priority_clf = SoftmaxLR(X.shape[1], len(PRIORITIES), seed=2).fit(X, yp)
        return self

    def evaluate(self, data):
        X = self.tfidf.transform([d["text"] for d in data])
        yi = np.array([INTENTS.index(d["intent"]) for d in data])
        yp = np.array([PRIORITIES.index(d["priority"]) for d in data])
        pi = self.intent_clf.predict_proba(X).argmax(1)
        pp = self.priority_clf.predict_proba(X).argmax(1)
        return {
            "intent_accuracy": float((pi == yi).mean()),
            "intent_macro_f1": macro_f1(yi, pi, range(len(INTENTS))),
            "priority_accuracy": float((pp == yp).mean()),
            "priority_macro_f1": macro_f1(yp, pp, range(len(PRIORITIES))),
            "n_test": int(len(data)),
        }

    URGENCY_SIGNALS = {"urgent", "urgently", "emergency", "asap", "immediately", "critical"}

    def predict(self, text):
        X = self.tfidf.transform([text])
        ip = self.intent_clf.predict_proba(X)[0]
        pp = self.priority_clf.predict_proba(X)[0]
        priority = PRIORITIES[int(pp.argmax())]
        source = "ml"
        # Hybrid rule layer (standard in production triage): explicit urgency
        # signals always escalate, even if the ML head is uncertain.
        if set(tokenize(text)) & self.URGENCY_SIGNALS:
            priority, source = "urgent", "rule_override"
        return {"intent": INTENTS[int(ip.argmax())], "intent_confidence": float(ip.max()),
                "priority": priority, "priority_confidence": float(pp.max()),
                "priority_source": source,
                "sentiment": sentiment(text)}

    def save(self, path):
        np.savez_compressed(path, vocab=json.dumps(self.tfidf.vocab), idf=self.tfidf.idf,
                            Wi=self.intent_clf.W, bi=self.intent_clf.b,
                            Wp=self.priority_clf.W, bp=self.priority_clf.b)

    @classmethod
    def load(cls, path):
        import os
        if not os.path.exists(path):
            return None
        m = cls(); z = np.load(path, allow_pickle=False)
        m.tfidf.vocab = json.loads(str(z["vocab"])); m.tfidf.idf = z["idf"]
        m.intent_clf = SoftmaxLR(z["Wi"].shape[0], len(INTENTS)); m.intent_clf.W, m.intent_clf.b = z["Wi"], z["bi"]
        m.priority_clf = SoftmaxLR(z["Wp"].shape[0], len(PRIORITIES)); m.priority_clf.W, m.priority_clf.b = z["Wp"], z["bp"]
        return m
