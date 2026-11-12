#!/usr/bin/env python3
"""Designed thesis-report PDF renderer (reportlab Platypus).

Usage:
    python3 build_designed_pdf.py <slug> [<slug> ...]
    python3 build_designed_pdf.py --all        # the 3 (later 6) known reports
    python3 build_designed_pdf.py agentforge-rag

Reads  <slug>-report.md  next to this script, plus the project's results.json
from ../<project-dir>/results.json for the charts, and writes <slug>-report.pdf.

Design system (see REPORT_SPEC.md "Design standard"):
  * full-page navy cover, per-project accent, headline-metric panel
  * "Results at a Glance" page: metric tiles + matplotlib chart(s)
  * styled TOC page (chapter list), running header/footer, accent chapter
    bands, zebra tables, grey code blocks, tinted callouts.
Content is NEVER altered: the markdown is the source of truth; this script
only re-presents it (cover / glance / TOC pages are presentation furniture
built from results.json values).
"""
import json, os, re, sys, textwrap

BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE, "assets")
os.makedirs(ASSETS, exist_ok=True)

NAVY = "0B1226"

CONFIGS = {
    "agentforge-rag": dict(
        project_dir="agentforge-rag", title="AgentForge-RAG", accent="2B50C8",
        short="AgentForge-RAG — Project Thesis Report",
        # Headline = REAL SciFact/BEIR results (results_real.json, Appendix C of
        # the report). The synthetic/internal 100% numbers are documented inside
        # as pipeline proof only and must not appear on the cover / glance
        # furniture. Groundedness / abstention are NOT benchmarked by SciFact
        # and likewise stay off the cover.
        metrics=[("0.6466", "nDCG@10 on SciFact test\nBEIR · 300 queries · real data"),
                 ("0.7059", "Recall@5 on SciFact test\nRecall@3 0.6578 · MRR@10 0.6090"),
                 ("0.665", "Published BEIR BM25 bar\nfinal sits below it — plateau"),
                 ("44 attempts", "Dev-screened improvement log\nRM3 reproduced harmful, rejected")],
        glance_caption="SciFact (BEIR) test results: the project's baseline ladder "
                       "(naive TF-IDF 0.5769 nDCG@10, hybrid 0.6269, hybrid + rerank "
                       "0.6429) and the dev-selected final pipeline (0.6466) against "
                       "the published BEIR BM25 bar of 0.665 (dashed). The final "
                       "pipeline — BM25 (k1=1.0, b=0.5) + TF-IDF title boost, RRF "
                       "fusion, no reranker — was selected on train/dev queries only "
                       "and evaluated on test exactly once; it plateaus below the "
                       "bar after 44 dev attempts. All values from "
                       "results_real.json; the main chapters' internal-eval results "
                       "stand as pipeline proof only."),
    "resolvedesk-ai": dict(
        project_dir="resolvedesk-ai", title="ResolveDesk AI", accent="4F46E5",
        short="ResolveDesk AI — Project Thesis Report",
        # Headline = REAL Bitext results (results_real.json, Appendix D of the report).
        # The synthetic-run numbers (incl. priority, which has no real labels and is
        # synthetic-only) are documented inside as pipeline proof only and must not
        # appear on the cover / glance furniture.
        metrics=[("97.59%", "Intent accuracy — Bitext deduped\nmacro-F1 0.9748, real data"),
                 ("99.5%", "Category accuracy — Bitext deduped\nmacro-F1 0.995, honest 2nd task"),
                 ("89.9%", "Deduped baseline intent accuracy\nproject classifier as-is"),
                 ("13/13", "Product tests passing\nincl. tenant isolation")],
        glance_caption="Deduped baseline vs final on the Bitext customer-support benchmark "
                       "(26,872 rows, 27 intents, licence CDLA-Sharing-1.0), deduplicated "
                       "split — the headline protocol (train 19,259 / test 4,813 unique "
                       "utterances). The naive split's 94.75% is leakage-inflated (13.79% of "
                       "its test rows appear verbatim in train) and is shown in Appendix D "
                       "as contrast only. Priority has no real labels and remains "
                       "synthetic-only, so it is not charted. All values from "
                       "results_real.json."),
    "sentinel-llm": dict(
        project_dir="sentinel-llm", title="Sentinel-LLM", accent="DC2626",
        short="Sentinel-LLM — Project Thesis Report",
        # Headline = REAL results (results_real.json Round 2, Appendix C of the
        # report). The synthetic-run numbers (100%, 191/191, 52.9x) are
        # documented inside as pipeline proof only and must not appear on the
        # cover / glance furniture.
        metrics=[("79.46%", "ATT&CK tactic test accuracy\nTF-IDF SVM · macro F1 0.7788"),
                 ("55.70%", "CVE severity test accuracy\nTF-IDF SVM · macro F1 0.5444"),
                 ("49.11% / 0.364", "Round 1 LoRA on ATT&CK\nlost to full FT — kept visible"),
                 ("81.95% / 0.7795", "Published ATT&CK SVM bar\ndifferent corpus / protocol")],
        glance_caption="Real-data test results (Round 2, Appendix C). Left: ATT&CK "
                       "tactic classification (stratified split, test n=112) — the "
                       "TF-IDF word 1-2-gram + linear SVM headline (79.46% accuracy, "
                       "0.7788 macro F1) is level with the published SVM bar on "
                       "macro F1 (0.7795) and 2.5 points below it on accuracy "
                       "(81.95%), on a different corpus/protocol; hybrid and neural "
                       "systems and the Round 1 LoRA final (49.11% / 0.364) sit "
                       "below it. Right: CVE severity (temporal split, test = 2024, "
                       "n=1,000) — headline 55.70% / 0.5444, below the literature "
                       "ceiling zone of about 70% at 36k-CVE scale. Hybrid variants "
                       "never beat pure TF-IDF. All values from results_real.json; "
                       "the main chapters' synthetic-data results stand as pipeline "
                       "proof only."),
    # --- future reports: same system, fill metrics from their results.json ---
    "densehar": dict(
        project_dir="densehar", title="DenseHAR", accent="7C3AED",
        short="DenseHAR — Project Thesis Report",
        # Headline = REAL HAPT results (results_real.json, Appendix C of the report).
        # The synthetic-run numbers are documented inside as pipeline proof only and
        # must not appear on the cover / glance furniture.
        metrics=[("90.38%", "Dense per-timestep accuracy\nHAPT · real data · 5 unseen subjects"),
                 ("0.9398", "Macro F1 on the HAPT test split\nwindow baseline 0.7258 (78.91% acc)"),
                 ("87.44%", "Transition-class timestep accuracy\ntransition-window 60.93% vs 47.87%"),
                 ("0.822 vs 0.749", "Segmental F1@25 on HAPT\nedit score 84.88 vs 75.52")],
        glance_caption="DenseHAR (final system: physics augmentation, 12-class supervision, "
                       "3-seed ensemble, val-tuned post-processing) vs the window-128 baseline "
                       "on HAPT — real data, subject-disjoint split (21 train / 4 validation / "
                       "5 unseen test subjects). Top: per-timestep metrics (macro F1 shown in "
                       "percent); the baseline's weakness concentrates exactly at transitions. "
                       "Bottom: segmentation quality — on real data DenseHAR also leads "
                       "segmental F1@25 and edit score, reversing the synthetic run's segmental "
                       "result. All values from results_real.json; the main chapters' "
                       "synthetic-data results stand as pipeline proof only."),
    "diffusionforge": dict(
        project_dir="diffusionforge", title="DiffusionForge", accent="DB2777",
        short="DiffusionForge — Project Thesis Report",
        # Headline = REAL MNIST-16 results (results_real.json, Appendix C of the report).
        # The synthetic-shapes numbers are documented inside as pipeline proof only and
        # must not appear on the cover / glance furniture.
        metrics=[("100.0%", "DDPM-300 class fidelity\nMNIST-16 real data · 70 of 70 samples"),
                 ("84.29%", "DDIM-50 class fidelity\nMNIST-16 · guidance w=1.5"),
                 ("0.0576", "Held-out noise-prediction MSE\nfinal model · real MNIST-16 test"),
                 ("98.13%", "Fidelity-classifier accuracy\nheld-out real test · gate PASS")],
        glance_caption="Real-data results on MNIST downscaled to 16x16 (Appendix C): the "
                       "baseline configuration and the validation-selected final model "
                       "(+12 epochs, no augmentation), each evaluated on test once at "
                       "guidance w=1.5. Left: class fidelity by sampler -- DDPM-300 against "
                       "DDIM-50; the dashed line is chance for 10 classes. Right: sample "
                       "diversity (mean pairwise L2) against real data. All values from "
                       "results_real.json; the main chapters' synthetic-shapes results "
                       "stand as pipeline proof only."),
    "fedguard": dict(
        project_dir="fedguard", title="FedGuard", accent="059669",
        short="FedGuard — Project Thesis Report",
        # Headline = REAL MNIST-16 results (results_real.json, Appendix C of the report).
        # The synthetic-run numbers are documented inside as pipeline proof only and
        # must not appear on the cover / glance furniture. Every epsilon-hat value is
        # the project's simplified RDP ESTIMATE, never a formal guarantee.
        metrics=[("97.56%", "Centralised accuracy\nMNIST-16 · full 10,000-image test"),
                 ("96.53%", "FedAvg non-IID accuracy\nDirichlet alpha=0.5 · 40 rounds"),
                 ("90.54%", "DP-SGD accuracy (sigma=1.2)\nepsilon-hat 5.55 (estimate)"),
                 ("92.18%", "Privacy-frontier maximum\nepsilon-hat 15.31 (estimate)")],
        glance_caption="Real-data results on MNIST downscaled to 16x16 (Appendix C), full "
                       "10,000-image test set for every row. Left: final test accuracy -- "
                       "centralised, FedAvg (IID and non-IID), FedProx (a negative result: "
                       "below FedAvg) and DP-SGD at two noise levels. Right: the "
                       "privacy-utility frontier; epsilon-hat values are the project's "
                       "simplified RDP estimates, not certified budgets. All values from "
                       "results_real.json; the main chapters' synthetic-data results stand "
                       "as pipeline proof only."),
}

# ----------------------------------------------------------------------------
# Charts (matplotlib, accent-matched)
# ----------------------------------------------------------------------------
def _mpl(accent):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": "#CBD5E1", "axes.linewidth": 0.8,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "text.color": "#0B1226", "axes.labelcolor": "#0B1226",
        "xtick.color": "#334155", "ytick.color": "#334155",
        "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt

def _save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path

def charts_for(slug, cfg):
    """Return list of PNG paths built from the project's results.json."""
    accent = "#" + cfg["accent"]
    rj = os.path.join(BASE, "..", cfg["project_dir"], "results.json")
    if not os.path.exists(rj):
        return []
    d = json.load(open(rj))
    plt = _mpl(accent)
    out = []
    if slug == "agentforge-rag":
        # Glance chart uses the REAL SciFact/BEIR results (results_real.json) —
        # the report's headline since Appendix C. The internal results.json
        # ablation values are not charted.
        rj_real = os.path.join(BASE, "..", cfg["project_dir"], "results_real.json")
        rd = json.load(open(rj_real))
        bt = rd["baseline_test"]; fin = rd["final_test"]["metrics"]
        bar = rd["published_bar"]["bm25_scifact_ndcg_at_10"]
        methods = ["Naive\nTF-IDF only", "Hybrid\nTF-IDF+BM25",
                   "Hybrid + rerank\n(best baseline rung)", "Final pipeline\n(dev-selected)"]
        ndcg = [bt["naive_tfidf_only"]["ndcg_at_10"], bt["hybrid_tfidf_bm25"]["ndcg_at_10"],
                bt["hybrid_plus_rerank"]["ndcg_at_10"], fin["ndcg_at_10"]]
        r5 = [bt["naive_tfidf_only"]["recall_at_5"], bt["hybrid_tfidf_bm25"]["recall_at_5"],
              bt["hybrid_plus_rerank"]["recall_at_5"], fin["recall_at_5"]]
        fig, ax = plt.subplots(figsize=(7.4, 3.9))
        x = range(4); w = 0.38
        b1 = ax.bar([i - w / 2 for i in x], ndcg, w, color=accent, label="nDCG@10")
        b2 = ax.bar([i + w / 2 for i in x], r5, w, color="#94A3B8", label="Recall@5")
        for b in list(b1) + list(b2):
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.axhline(bar, color="#DC2626", lw=1.2, ls="--")
        ax.annotate(f"published BEIR BM25 bar (nDCG@10 {bar})", (3.45, 0.815),
                    ha="right", fontsize=8, color="#DC2626")
        ax.set_xticks(list(x)); ax.set_xticklabels(methods, fontsize=8.5)
        ax.set_ylim(0, 0.88); ax.set_ylabel("Score")
        ax.set_title("SciFact (BEIR) test — baseline ladder vs final (real data, 300 queries)", fontsize=11)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2); ax.grid(axis="y", color="#E2E8F0", lw=0.7)
        ax.set_axisbelow(True)
        out.append(_save(fig, "agentforge-ablation.png"))
    elif slug == "resolvedesk-ai":
        # Glance chart uses the REAL Bitext results (results_real.json) — the report's
        # headline since Appendix D. The synthetic results.json values (incl. priority)
        # are not charted.
        rj_real = os.path.join(BASE, "..", cfg["project_dir"], "results_real.json")
        rd = json.load(open(rj_real))
        bl = rd["baseline"]["deduped"]; fn = rd["final"]
        labels = ["Intent\naccuracy", "Intent\nmacro-F1", "Category\naccuracy", "Category\nmacro-F1"]
        base_vals = [bl["intent"]["accuracy"] * 100, bl["intent"]["macro_f1"] * 100,
                     bl["category"]["accuracy"] * 100, bl["category"]["macro_f1"] * 100]
        final_vals = [fn["intent"]["accuracy"] * 100, fn["intent"]["macro_f1"] * 100,
                      fn["category"]["accuracy"] * 100, fn["category"]["macro_f1"] * 100]
        fig, ax = plt.subplots(figsize=(7.4, 3.9))
        x = range(4); w = 0.38
        b1 = ax.bar([i - w / 2 for i in x], base_vals, w, color="#94A3B8", label="Deduped baseline")
        b2 = ax.bar([i + w / 2 for i in x], final_vals, w, color=accent, label="Final (val-selected)")
        for b in list(b1) + list(b2):
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 112); ax.set_ylabel("Score (%)")
        ax.set_title("Bitext deduped test — baseline vs final (real data, n=4,813)", fontsize=11)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.13), ncol=2)
        ax.grid(axis="y", color="#E2E8F0", lw=0.7); ax.set_axisbelow(True)
        out.append(_save(fig, "resolvedesk-classifier.png"))
    elif slug == "sentinel-llm":
        # Glance charts use the REAL results (results_real.json Round 2) — the
        # report's headline since Appendix C. The synthetic results.json values
        # are not charted.
        rj_real = os.path.join(BASE, "..", cfg["project_dir"], "results_real.json")
        rd = json.load(open(rj_real))
        systems = ["TF-IDF SVM\n(headline)", "Hybrid", "Neural best\n(full FT)", "LoRA\n(Round 1 final)"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.8))
        for ax, task, ttl in ((a1, "mitre", "ATT&CK tactic — test (n=112)"),
                              (a2, "cve", "CVE severity — test 2024 (n=1,000)")):
            fam = rd["tasks"][task]["round2"]["families"]
            head = rd["round2"]["headlines"][task]["test"]
            lora = rd["tasks"][task]["phase_b_final"]["lora_test"]
            acc = [head["accuracy"] * 100, fam["hybrid"]["winner_test"]["accuracy"] * 100,
                   fam["neural"]["winner_test"]["accuracy"] * 100, lora["accuracy"] * 100]
            f1 = [head["macro_f1"] * 100, fam["hybrid"]["winner_test"]["macro_f1"] * 100,
                  fam["neural"]["winner_test"]["macro_f1"] * 100, lora["macro_f1"] * 100]
            x = range(4); w = 0.38
            b1 = ax.bar([i - w / 2 for i in x], acc, w, color=accent, label="Accuracy")
            b2 = ax.bar([i + w / 2 for i in x], f1, w, color="#94A3B8", label="Macro F1")
            for b in list(b1) + list(b2):
                ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=6.8, fontweight="bold")
            ax.set_xticks(list(x)); ax.set_xticklabels(systems, fontsize=7.5)
            ax.set_ylim(0, 100); ax.set_ylabel("Score (%)")
            ax.set_title(ttl, fontsize=10.5)
            ax.grid(axis="y", color="#E2E8F0", lw=0.7); ax.set_axisbelow(True)
        a1.legend(frameon=False, fontsize=8, loc="upper center",
                  bbox_to_anchor=(1.0, -0.13), ncol=2)
        fig.tight_layout()
        out.append(_save(fig, "sentinel-tasks-params.png"))
    elif slug == "densehar":
        # Glance charts use the REAL HAPT results (results_real.json) — the report's
        # headline since Appendix C. The synthetic results.json values are not charted.
        rj_real = os.path.join(BASE, "..", cfg["project_dir"], "results_real.json")
        rd = json.load(open(rj_real))
        fh = rd["final"]["headline_test"]; bl = rd["final"]["baseline_test"]
        labels = ["Overall\naccuracy", "Macro F1", "Transition-window\naccuracy",
                  "Transition-class\naccuracy"]
        dense_vals = [fh["accuracy"] * 100, fh["macro_f1"] * 100,
                      fh["transition_window_accuracy"] * 100,
                      fh["transition_class_timestep_accuracy"] * 100]
        base_vals = [bl["accuracy"] * 100, bl["macro_f1"] * 100,
                     bl["transition_window_accuracy"] * 100,
                     bl["transition_class_timestep_accuracy"] * 100]
        fig = plt.figure(figsize=(7.6, 4.9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.62, wspace=0.22)
        a1 = fig.add_subplot(gs[0, :]); a2 = fig.add_subplot(gs[1, 0]); a3 = fig.add_subplot(gs[1, 1])
        x = range(4); w = 0.38
        b1 = a1.bar([i - w / 2 for i in x], dense_vals, w, color=accent, label="DenseHAR")
        b2 = a1.bar([i + w / 2 for i in x], base_vals, w, color="#94A3B8", label="Window baseline")
        for b in list(b1) + list(b2):
            a1.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        a1.set_xticks(list(x)); a1.set_xticklabels(labels, fontsize=8.5)
        a1.set_ylim(0, 112); a1.set_ylabel("Score (%)")
        a1.set_title("DenseHAR vs window baseline — HAPT test (real data)", fontsize=11)
        a1.legend(frameon=False, fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=2)
        a1.grid(axis="y", color="#E2E8F0", lw=0.7); a1.set_axisbelow(True)
        for ax, key, ttl, ylab, fmt, ymax in (
                (a2, "segmental_f1@25", "Segmental F1@25", "Segmental F1", "{:.3f}", 1.0),
                (a3, "edit_score", "Edit score", "Edit score", "{:.1f}", 100)):
            bars = ax.bar(["DenseHAR", "Baseline"], [fh[key], bl[key]], 0.5,
                          color=[accent, "#94A3B8"])
            for b in bars:
                ax.annotate(fmt.format(b.get_height()),
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_ylim(0, ymax * 1.15); ax.set_ylabel(ylab, fontsize=9)
            ax.set_title(ttl, fontsize=11)
            ax.tick_params(axis="x", labelsize=8.5)
            ax.grid(axis="y", color="#E2E8F0", lw=0.7); ax.set_axisbelow(True)
        out.append(_save(fig, "densehar-transition-segmental.png"))
    elif slug == "diffusionforge":
        # Glance charts use the REAL MNIST-16 results (results_real.json) — the
        # report's headline since Appendix C. The synthetic results.json values
        # are not charted.
        rj_real = os.path.join(BASE, "..", cfg["project_dir"], "results_real.json")
        rd = json.load(open(rj_real))
        bt = rd["baseline"]["test_evaluation"]; ft = rd["final"]["test_evaluation"]
        labels = ["Baseline\nDDPM-300", "Baseline\nDDIM-50", "Final\nDDPM-300", "Final\nDDIM-50"]
        vals = [bt["ddpm"]["fidelity_pct"], bt["ddim50"]["fidelity_pct"],
                ft["ddpm"]["fidelity_pct"], ft["ddim50"]["fidelity_pct"]]
        cols = ["#94A3B8", "#94A3B8", accent, accent]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.8),
                                     gridspec_kw={"width_ratios": [1.9, 1]})
        bars = a1.bar(labels, vals, 0.55, color=cols)
        for b in bars:
            a1.annotate(f"{b.get_height():.2f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        a1.axhline(10, color="#DC2626", lw=1, ls="--")
        a1.annotate("chance (10%)", (3.45, 12), ha="right", fontsize=8, color="#DC2626")
        a1.set_ylim(0, 115); a1.set_ylabel("Class fidelity (%)")
        a1.set_title("Fidelity on MNIST-16: baseline vs final", fontsize=11)
        a1.tick_params(axis="x", labelsize=8)
        a1.grid(axis="y", color="#E2E8F0", lw=0.7); a1.set_axisbelow(True)
        dv = a2.bar(["DDPM", "DDIM", "Real\ndata"],
                    [ft["ddpm"]["diversity_pairwise_l2"], ft["ddim50"]["diversity_pairwise_l2"],
                     ft["real_data_diversity_pairwise_l2"]],
                    0.5, color=[accent, "#334155", "#94A3B8"])
        for b in dv:
            a2.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        a2.set_ylim(0, 11.5); a2.set_ylabel("Mean pairwise L2")
        a2.set_title("Final diversity vs real", fontsize=11)
        a2.tick_params(axis="x", labelsize=8)
        a2.grid(axis="y", color="#E2E8F0", lw=0.7); a2.set_axisbelow(True)
        fig.tight_layout()
        out.append(_save(fig, "diffusionforge-fidelity-speed.png"))
    elif slug == "fedguard":
        # Glance charts use the REAL MNIST-16 results (results_real.json) — the
        # report's headline since Appendix C. The synthetic results.json values
        # are not charted. Epsilon-hat is the project's simplified RDP ESTIMATE.
        rj_real = os.path.join(BASE, "..", cfg["project_dir"], "results_real.json")
        rd = json.load(open(rj_real))
        fin = rd["final"]
        keys = ["centralised", "fedavg_iid", "fedavg_noniid", "fedprox_noniid",
                "dp_sigma1.2", "dp_sigma0.7"]
        labels = ["Centralised\n(pooled)", "FedAvg\nIID", "FedAvg\nnon-IID",
                  "FedProx\nnon-IID", "DP\ns=1.2", "DP\ns=0.7"]
        vals = [fin[k]["test_accuracy"] * 100 for k in keys]
        cols = ["#94A3B8", accent, accent, "#94A3B8", "#334155", "#334155"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.9),
                                     gridspec_kw={"width_ratios": [1.85, 1]})
        bars = a1.bar(labels, vals, 0.58, color=cols)
        for b in bars:
            a1.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
        a1.set_ylim(0, 108); a1.set_ylabel("Test accuracy (%)")
        a1.set_title("Final accuracy — MNIST-16 (real data)", fontsize=11)
        a1.tick_params(axis="x", labelsize=8)
        a1.grid(axis="y", color="#E2E8F0", lw=0.7); a1.set_axisbelow(True)
        fr = rd["privacy_frontier"]
        eps = [p["epsilon_estimate"] for p in fr]
        acc = [p["test_accuracy"] * 100 for p in fr]
        a2.axhline(fin["fedavg_noniid"]["test_accuracy"] * 100, color="#94A3B8",
                   lw=1, ls="--")
        a2.annotate("dashed: FedAvg non-IID 96.5\n(no DP)", (0.04, 0.90),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=7.5, color="#64748B")
        a2.scatter(eps, acc, s=70, color=accent, zorder=3)
        a2.plot(eps, acc, color=accent, lw=1, alpha=0.55, zorder=2)
        for e, a in zip(eps, acc):
            a2.annotate(f"{a:.1f}%", (e, a), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, fontweight="bold")
        a2.set_xlim(2.0, 17.5); a2.set_ylim(74, 100)
        a2.set_xlabel("Epsilon-hat (estimate)")
        a2.set_ylabel("Test accuracy (%)")
        a2.set_title("Privacy vs utility (frontier)", fontsize=11)
        a2.tick_params(axis="x", labelsize=8)
        a2.grid(color="#E2E8F0", lw=0.7); a2.set_axisbelow(True)
        fig.tight_layout()
        out.append(_save(fig, "fedguard-accuracy-privacy.png"))
    return out

# ----------------------------------------------------------------------------
# Fonts & colour helpers
# ----------------------------------------------------------------------------
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

_F = "/usr/share/fonts/truetype/noto/"
pdfmetrics.registerFont(TTFont("Noto", _F + "NotoSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Noto-B", _F + "NotoSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Noto-I", _F + "NotoSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Noto-BI", _F + "NotoSans-BoldItalic.ttf"))
pdfmetrics.registerFont(TTFont("Mono", _F + "NotoSansMono-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Mono-B", _F + "NotoSansMono-Bold.ttf"))
pdfmetrics.registerFontFamily("Noto", normal="Noto", bold="Noto-B",
                              italic="Noto-I", boldItalic="Noto-BI")
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-B",
                              italic="Mono", boldItalic="Mono-B")

def C(hexs): return colors.HexColor("#" + hexs)

def lighten(hexs, f=0.72):
    c = C(hexs)
    return colors.Color(1 - (1 - c.red) * (1 - f), 1 - (1 - c.green) * (1 - f),
                        1 - (1 - c.blue) * (1 - f))
def darken(hexs, f=0.35):
    c = C(hexs)
    return colors.Color(c.red * (1 - f), c.green * (1 - f), c.blue * (1 - f))

# ----------------------------------------------------------------------------
# Inline markdown -> reportlab markup
# ----------------------------------------------------------------------------
def inline(text, code_color="7C2D12"):
    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    codes = []
    def stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes)-1}\x00"
    t = re.sub(r"`([^`]+)`", stash, t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    def unstash(m):
        return f'<font face="Mono" size="8.6" color="#{code_color}">{codes[int(m.group(1))]}</font>'
    return re.sub("\x00(\\d+)\x00", unstash, t)

# ----------------------------------------------------------------------------
# Markdown -> flowables
# ----------------------------------------------------------------------------
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, PageBreak, NextPageTemplate, Table,
                                TableStyle, Image, HRFlowable, Preformatted,
                                KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

CHAPTER_STARTS = []   # (title, page) filled during build pass 1

def build_styles(accent):
    A = C(accent); NAV = C(NAVY)
    s = {}
    s["body"] = ParagraphStyle("body", fontName="Noto", fontSize=10.5, leading=15.6,
                               textColor=C("1E293B"), spaceAfter=6, alignment=TA_LEFT)
    s["bodyc"] = ParagraphStyle("bodyc", parent=s["body"], alignment=TA_CENTER)
    s["chapter"] = ParagraphStyle("chapter", fontName="Noto-B", fontSize=20, leading=25,
                                  textColor=NAV, spaceBefore=2, spaceAfter=4,
                                  keepWithNext=1)
    s["section"] = ParagraphStyle("section", fontName="Noto-B", fontSize=13.5, leading=18,
                                  textColor=darken(accent, 0.12), spaceBefore=13,
                                  spaceAfter=5, keepWithNext=1)
    s["sub"] = ParagraphStyle("sub", fontName="Noto-B", fontSize=11.5, leading=15,
                              textColor=C("0F172A"), spaceBefore=10, spaceAfter=4,
                              keepWithNext=1)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=14,
                                 bulletIndent=3, spaceAfter=3.5)
    s["cell"] = ParagraphStyle("cell", fontName="Noto", fontSize=8.2, leading=11,
                               textColor=C("1E293B"))
    s["cellh"] = ParagraphStyle("cellh", fontName="Noto-B", fontSize=8.4, leading=11,
                                textColor=colors.white)
    s["code"] = ParagraphStyle("code", fontName="Mono", fontSize=7.4, leading=10.2,
                               textColor=C("0F172A"), backColor=C("F1F5F9"),
                               borderPadding=(6, 8, 6), spaceBefore=3, spaceAfter=3)
    s["quote"] = ParagraphStyle("quote", parent=s["body"], textColor=C("334155"),
                                fontName="Noto-I")
    s["toc1"] = ParagraphStyle("toc1", fontName="Noto-B", fontSize=11.5, leading=15,
                               textColor=NAV, spaceBefore=7)
    s["toc2"] = ParagraphStyle("toc2", fontName="Noto", fontSize=10, leading=14,
                               textColor=C("475569"), leftIndent=16, spaceBefore=2)
    s["caption"] = ParagraphStyle("caption", fontName="Noto-I", fontSize=9, leading=13,
                                  textColor=C("64748B"), alignment=TA_CENTER, spaceBefore=4)
    s["glanceh"] = ParagraphStyle("glanceh", fontName="Noto-B", fontSize=21, leading=26,
                                  textColor=NAV, spaceAfter=2)
    return s

def callout(flowables, accent, S):
    inner = [[flowables]]
    t = Table(inner, colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), lighten(accent, 0.88)),
        ("LINEBEFORE", (0, 0), (0, -1), 3, C(accent)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t

def md_to_flowables(lines, S, accent, chapters_out):
    fl = []
    i, n = 0, len(lines)
    para = []
    def flush():
        if para:
            txt = " ".join(x.strip() for x in para).strip()
            if txt:
                fl.append(Paragraph(inline(txt), S["body"]))
            para.clear()
    while i < n:
        line = lines[i]; stripped = line.strip()
        # fenced code
        if stripped.startswith("```"):
            flush(); i += 1; buf = []
            while i < n:
                s3 = lines[i].strip()
                if s3 == "```":
                    i += 1; break
                if s3.endswith("```") and len(s3) > 3 and not s3.startswith("```"):
                    buf.append(s3[:-3]); i += 1; break
                buf.append(lines[i]); i += 1
            wrapped = []
            for b in buf:
                wrapped.extend(textwrap.wrap(b, 96, replace_whitespace=False,
                                             drop_whitespace=False) or [""])
            pf = Preformatted("\n".join(wrapped), S["code"], maxLineLength=96)
            fl.append(pf); fl.append(Spacer(1, 4))
            continue
        # table
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            flush()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            header, body = rows[0], rows[2:]
            ncol = len(header)
            widths_src = header + (body[0] if body else [])
            lens = [max(6, max(len(r[j]) if j < len(r) else 6 for r in rows)) for j in range(ncol)]
            tot = sum(lens); total_w = 174 * mm
            widths = [max(18 * mm, total_w * l / tot) for l in lens]
            scale = total_w / sum(widths); widths = [w * scale for w in widths]
            data = [[Paragraph(inline(c), S["cellh"]) for c in header]]
            for r in body:
                r = (r + [""] * ncol)[:ncol]
                data.append([Paragraph(inline(c), S["cell"]) for c in r])
            t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            style = [("BACKGROUND", (0, 0), (-1, 0), C(accent)),
                     ("GRID", (0, 0), (-1, -1), 0.4, C("CBD5E1")),
                     ("VALIGN", (0, 0), (-1, -1), "TOP"),
                     ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                     ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
            for r in range(1, len(data)):
                if r % 2 == 0:
                    style.append(("BACKGROUND", (0, r), (-1, r), lighten(accent, 0.93)))
            t.setStyle(TableStyle(style))
            fl.append(t); fl.append(Spacer(1, 6))
            continue
        # standalone image: ![alt](path-relative-to-reports-dir)
        mi = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if mi:
            flush()
            img_path = os.path.join(BASE, mi.group(2).strip())
            if os.path.exists(img_path):
                from PIL import Image as _PILImage
                iw, ih = _PILImage.open(img_path).size
                w = 148 * mm; h = w * ih / iw
                if h > 82 * mm:
                    h = 82 * mm; w = h * iw / ih
                fl.append(Image(img_path, width=w, height=h, hAlign="CENTER"))
                if mi.group(1).strip():
                    fl.append(Paragraph(inline(mi.group(1).strip()), S["caption"]))
                fl.append(Spacer(1, 6))
            i += 1; continue
        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m and not stripped.startswith("# " * 2):
            flush()
            level, text = len(m.group(1)), m.group(2).strip()
            text = re.sub(r"\*\*", "", text)
            if level == 1:
                chapters_out.append(text)
                fl.append(PageBreak())
                fl.append(Paragraph(inline(text), S["chapter"], bulletText=None))
                fl.append(HRFlowable(width="100%", thickness=2.2, color=C(accent),
                                     spaceBefore=1, spaceAfter=9))
            elif level == 2:
                fl.append(Paragraph(inline(text), S["section"]))
            else:
                fl.append(Paragraph(inline(text), S["sub"]))
            i += 1; continue
        # hr
        if re.match(r"^---+$", stripped):
            flush(); fl.append(HRFlowable(width="100%", thickness=0.7,
                                          color=C("CBD5E1"), spaceBefore=5, spaceAfter=8))
            i += 1; continue
        # blockquote
        if stripped.startswith(">"):
            flush(); buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip()); i += 1
            fl.append(callout(Paragraph(inline(" ".join(buf)), S["quote"]), accent, S))
            fl.append(Spacer(1, 5)); continue
        # lists
        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+[.)]\s+", stripped):
            flush()
            while i < n:
                s2 = lines[i].strip()
                mb = re.match(r"^[-*]\s+(.*)$", s2); mn = re.match(r"^(\d+)[.)]\s+(.*)$", s2)
                if mb:
                    fl.append(Paragraph(f'<font color="#{accent}">\u25aa</font>&nbsp;&nbsp;'
                                        + inline(mb.group(1)), S["bullet"]))
                elif mn:
                    fl.append(Paragraph(f'<font color="#{accent}"><b>{mn.group(1)}.</b></font>&nbsp;&nbsp;'
                                        + inline(mn.group(2)), S["bullet"]))
                else:
                    break
                i += 1
            fl.append(Spacer(1, 3)); continue
        if stripped == "":
            flush(); i += 1; continue
        para.append(line); i += 1
    flush()
    return fl

# ----------------------------------------------------------------------------
# Cover + page furniture
# ----------------------------------------------------------------------------
def wrap_text(text, font, size, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if pdfmetrics.stringWidth(t, font, size) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def draw_cover(canvas, doc, cfg, subtitle):
    canvas.saveState()
    W, H = A4
    A = C(cfg["accent"]); NAV = C(NAVY); light = lighten(cfg["accent"], 0.35)
    canvas.setFillColor(NAV); canvas.rect(0, 0, W, H, stroke=0, fill=1)
    # geometric accents
    canvas.setFillColor(A)
    canvas.saveState(); canvas.translate(W - 40, H - 30); canvas.rotate(24)
    canvas.rect(-30, -260, 88, 340, stroke=0, fill=1); canvas.restoreState()
    canvas.setFillColor(lighten(cfg["accent"], 0.45))
    canvas.saveState(); canvas.translate(W - 40, H - 30); canvas.rotate(24)
    canvas.rect(66, -260, 16, 340, stroke=0, fill=1); canvas.restoreState()
    canvas.setStrokeColor(A); canvas.setLineWidth(1.1)
    canvas.circle(-70, 60, 150, stroke=1, fill=0)
    canvas.setStrokeColor(lighten(cfg["accent"], 0.3))
    canvas.circle(-70, 60, 118, stroke=1, fill=0)
    x = 54
    # top label
    canvas.setFillColor(light); canvas.setFont("Noto-B", 10.5)
    canvas.drawString(x, H - 74, "P R O J E C T   T H E S I S   R E P O R T")
    canvas.setFillColor(A); canvas.rect(x, H - 86, 54, 3, stroke=0, fill=1)
    # title
    y = H - 168
    canvas.setFillColor(colors.white)
    for ln in wrap_text(cfg["title"], "Noto-B", 40, W - 2 * x - 60):
        canvas.setFont("Noto-B", 40); canvas.drawString(x, y, ln); y -= 48
    y -= 6
    canvas.setFillColor(light)
    for ln in wrap_text(subtitle, "Noto", 14.5, W - 2 * x - 40)[:5]:
        canvas.setFont("Noto", 14.5); canvas.drawString(x, y, ln); y -= 21
    # headline metrics panel
    metrics = cfg.get("metrics") or []
    if metrics:
        panel_y = 268
        canvas.setFillColor(colors.HexColor("#16233F"))
        canvas.setStrokeColor(A); canvas.setLineWidth(1)
        canvas.roundRect(x, panel_y, W - 2 * x, 168, 10, stroke=1, fill=1)
        canvas.setFillColor(light); canvas.setFont("Noto-B", 9.5)
        canvas.drawString(x + 18, panel_y + 146, "H E A D L I N E   R E S U L T S")
        colw = (W - 2 * x - 36) / 2
        for idx, (val, lab) in enumerate(metrics[:4]):
            cx = x + 18 + (idx % 2) * colw
            cy = panel_y + 108 - (idx // 2) * 56
            canvas.setFillColor(lighten(cfg["accent"], 0.22)); canvas.setFont("Noto-B", 21)
            canvas.drawString(cx, cy, val)
            canvas.setFillColor(colors.HexColor("#B9C6DE")); canvas.setFont("Noto", 8.6)
            for j, ll in enumerate(lab.split("\n")):
                canvas.drawString(cx, cy - 14 - j * 11, ll)
    # author block
    canvas.setFillColor(A); canvas.rect(x, 118, 54, 3, stroke=0, fill=1)
    canvas.setFillColor(colors.white); canvas.setFont("Noto-B", 15)
    canvas.drawString(x, 96, "Ashutosh Sahoo")
    canvas.setFillColor(colors.HexColor("#CBD5E1")); canvas.setFont("Noto", 10.5)
    canvas.drawString(x, 79, "M.Tech, Artificial Intelligence  ·  NIT Silchar")
    canvas.drawString(x, 63, "November 2026")
    canvas.restoreState()

class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, cfg, chapter_pages, **kw):
        super().__init__(filename, pagesize=A4, **kw)
        self.cfg = cfg; self.chapter_pages = chapter_pages
        self.current_chapter = ""
        W, H = A4
        full = Frame(0, 0, W, H, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        body = Frame(18 * mm, 17 * mm, W - 36 * mm, H - 34 * mm,
                     leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="Cover", frames=[full], onPage=self._cover),
            PageTemplate(id="Body", frames=[body], onPage=self._furniture)])
    def _cover(self, canvas, doc):
        draw_cover(canvas, doc, self.cfg, self.cfg["_subtitle"])
    def _furniture(self, canvas, doc):
        canvas.saveState()
        W, H = A4; A = C(self.cfg["accent"])
        chapter = self.current_chapter
        for title, pg in self.chapter_pages:
            if pg <= doc.page: chapter = title
        canvas.setFont("Noto-B", 8); canvas.setFillColor(C("0F172A"))
        canvas.drawString(18 * mm, H - 12 * mm, self.cfg["short"].split(" — ")[0].upper())
        canvas.setFont("Noto", 8); canvas.setFillColor(C("64748B"))
        canvas.drawRightString(W - 18 * mm, H - 12 * mm, chapter[:64])
        canvas.setStrokeColor(A); canvas.setLineWidth(1.4)
        canvas.line(18 * mm, H - 14.2 * mm, W - 18 * mm, H - 14.2 * mm)
        canvas.setStrokeColor(C("E2E8F0")); canvas.setLineWidth(0.6)
        canvas.line(18 * mm, 13.5 * mm, W - 18 * mm, 13.5 * mm)
        canvas.setFont("Noto", 8); canvas.setFillColor(C("64748B"))
        canvas.drawString(18 * mm, 9 * mm, "Ashutosh Sahoo · M.Tech (AI), NIT Silchar")
        canvas.setFont("Noto-B", 8.5); canvas.setFillColor(C("0F172A"))
        canvas.drawRightString(W - 18 * mm, 9 * mm, f"{doc.page}")
        canvas.restoreState()
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            st = flowable.style.name
            if st == "chapter":
                txt = flowable.getPlainText()
                self.current_chapter = txt
                CHAPTER_STARTS.append((txt, self.page))

# ----------------------------------------------------------------------------
# Build one report
# ----------------------------------------------------------------------------
def extract_title_block(lines):
    """Return (subtitle, start_index): content begins at the Abstract heading."""
    subtitle = ""
    for i, ln in enumerate(lines[:8]):
        if ln.startswith("## "):
            subtitle = ln[3:].strip()
        if ln.startswith("### ") and subtitle:
            subtitle += " — " + ln[4:].strip()
    start = 0
    for i, ln in enumerate(lines):
        if re.match(r"^#{1,2}\s+Abstract\b", ln.strip()):
            start = i; break
    return subtitle, start

def glance_flowables(cfg, S, chart_paths):
    fl = [Paragraph("Results at a Glance", S["glanceh"]),
          HRFlowable(width="100%", thickness=2.2, color=C(cfg["accent"]),
                     spaceBefore=1, spaceAfter=10)]
    metrics = cfg.get("metrics") or []
    if metrics:
        cells = []
        for val, lab in metrics[:4]:
            lab_html = "<br/>".join(lab.split("\n"))
            cells.append(Paragraph(
                f'<font face="Noto-B" size="19" color="#{cfg["accent"]}">{val}</font>'
                f'<br/><font size="8.4" color="#475569">{lab_html}</font>', S["body"]))
        while len(cells) % 2: cells.append(Paragraph("", S["body"]))
        rows = [cells[i:i + 2] for i in range(0, len(cells), 2)]
        t = Table(rows, colWidths=[87 * mm, 87 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), lighten(cfg["accent"], 0.94)),
            ("BOX", (0, 0), (-1, -1), 0.6, C("CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, C("CBD5E1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        fl.append(t); fl.append(Spacer(1, 10))
    for p in chart_paths:
        from PIL import Image as PILImage
        iw, ih = PILImage.open(p).size
        w = 168 * mm; h = w * ih / iw
        if h > 92 * mm:
            h = 92 * mm; w = h * iw / ih
        fl.append(Image(p, width=w, height=h, hAlign="CENTER"))
    if cfg.get("glance_caption"):
        fl.append(Paragraph(cfg["glance_caption"], S["caption"]))
    return fl

def build(slug):
    cfg = dict(CONFIGS[slug]); accent = cfg["accent"]
    md_path = os.path.join(BASE, f"{slug}-report.md")
    out_path = os.path.join(BASE, f"{slug}-report.pdf")
    lines = open(md_path, encoding="utf-8").read().splitlines()
    subtitle, start = extract_title_block(lines)
    cfg["_subtitle"] = subtitle or cfg["title"]
    S = build_styles(accent)
    chart_paths = charts_for(slug, cfg)

    # drop the md's own TOC section (we render a designed one)
    body_lines, skip = [], False
    for ln in lines[start:]:
        if re.match(r"^#{1,2}\s+Table of Contents", ln.strip()):
            skip = True; continue
        if skip and (ln.startswith("# ") or re.match(r"^#{1,2}\s+", ln.strip())):
            skip = False
        if not skip: body_lines.append(ln)

    chapters = []
    _ = md_to_flowables(body_lines, S, accent, chapters)   # collect chapter list

    def make_story():
        toc_items = ["Abstract"] + chapters
        toc_fl = [Paragraph("Contents", S["glanceh"]),
                  HRFlowable(width="100%", thickness=2.2, color=C(accent),
                             spaceBefore=1, spaceAfter=8)]
        for c in toc_items:
            style = S["toc1"] if (c == "Abstract" or re.match(r"^\d+\.|^Appendix|^Chapter", c)) else S["toc2"]
            toc_fl.append(Paragraph(inline(re.sub(r"\*\*", "", c)), style))
        content = md_to_flowables(body_lines, S, accent, [])
        story = [NextPageTemplate("Body"), PageBreak()]                 # cover
        story += glance_flowables(cfg, S, chart_paths)                  # glance
        story += [PageBreak()] + toc_fl                                 # contents
        story += [PageBreak()] + content                                # chapters
        return story

    global CHAPTER_STARTS
    snapshot = []
    for attempt in (1, 2):
        CHAPTER_STARTS = []
        doc = ReportDoc(out_path if attempt == 2 else out_path + ".tmp",
                        cfg, snapshot)
        doc.build(make_story())
        if attempt == 1:
            snapshot = list(CHAPTER_STARTS)
    os.path.exists(out_path + ".tmp") and os.remove(out_path + ".tmp")
    from pypdf import PdfReader
    return len(PdfReader(out_path).pages), out_path

if __name__ == "__main__":
    args = sys.argv[1:]
    slugs = ["agentforge-rag", "resolvedesk-ai", "sentinel-llm"] if (not args or args == ["--all"]) else args
    for s in slugs:
        pages, path = build(s)
        print(f"{s}: {pages} pages -> {path}")
