# LinkedIn Posts — Ashutosh Sahoo
### 5 ready-to-post announcements (November 2026)

Post (a) first. Space (b), (c) and (d) a few days apart, and post (e) last. Numbers in (a)–(c) are from the
projects' `results.json` files, measured on each project's built-in evaluation data;
post (d) reports DenseHAR's **real-data HAPT results** (`results_real.json`) and
supersedes the DenseHAR figures in (a) and (c). Post (e) reports the **full six-project real-benchmark programme** (every project's `results_real.json`; summary in `REAL_BENCHMARKS.md`) and supersedes the synthetic headlines in (a)–(c) wherever they differ. If anyone asks in comments, say
exactly that; it is the strongest possible answer.

---

## (a) The 6-Project Build — Launch Post

Over the past weeks I set myself a specific brief: build six complete AI projects, each one runnable end-to-end, each one evaluated with baselines and ablations, and each one documented with a full written report. They are done, and every number below was measured, not targeted.

1. Sentinel-LLM — LoRA fine-tuning for cybersecurity. I implemented LoRA from scratch in PyTorch; the fine-tuned model matched full fine-tuning (100% on the held-out test set) while training 1.86% of the parameters.

2. DenseHAR — dense, per-timestep activity recognition that handles transitions between activities, extending my InceptionKAN research. 95.35% accuracy on unseen subjects, and 90.63% inside transition windows where a standard windowed baseline scored 48.03%.

3. AgentForge-RAG — an agentic RAG system with hybrid retrieval (TF-IDF + BM25 + reranking) and guardrails. Recall@5 of 100% on both its eval splits, with groundedness, abstention and prompt-injection block rates also at 100% on their test sets.

4. DiffusionForge — diffusion models from scratch (DDPM, DDIM, classifier-free guidance). Generated samples reached 98.44% class fidelity; DDIM sampling was 6.5× faster than full DDPM sampling.

5. FedGuard — federated learning with differential privacy. FedProx recovered accuracy lost to non-IID client data (87.5% → 88.5%, centralised bound 89.0%), and DP-SGD training quantified what privacy costs: about 5 accuracy points at a single-digit estimated epsilon.

6. ResolveDesk AI — the product of the set: a working multi-tenant AI support desk for small businesses, with cited answers, automatic ticket triage and deflection analytics.

Two honest notes, because I think they matter more than the scores: each result was measured on that project's own built-in evaluation data (the protocols are documented in each repo), and one project produced a counter-result — my dense model over-segments, and a simpler baseline beats it on segmental F1. That finding is written up in its report, with the fix I would implement next.

Code, results files and 34–45 page reports for all six are organised in one portfolio. I am now open to AI Engineer roles and freelance work in LLM fine-tuning, RAG systems and custom ML development — if your team is hiring or your business needs an AI system built and honestly evaluated, my inbox is open.

GitHub: github.com/Ashutosh130801

#MachineLearning #DeepLearning #LLM #FineTuning #RAG #PyTorch #FederatedLearning #DiffusionModels #HumanActivityRecognition #AIEngineering

---

## (b) Sentinel-LLM Deep-Dive Post

A detail from my Sentinel-LLM project that I keep thinking about, because it changed how I read fine-tuning results:

I ran a sweep of four LoRA configurations (rank 4, 8 and 16, at two learning rates) on a cybersecurity instruction dataset — six tasks including MITRE ATT&CK mapping, log triage and phishing detection. Every single configuration reached the same validation accuracy: 99.44%. If I had reported accuracy alone, the sweep would have looked pointless.

The differences were in the validation loss: 0.0475 at rank 4, 0.0230 at rank 8, 0.0065 at rank 16. Rank was not buying correctness on this data; it was buying confidence — sharper, better-calibrated predictions. At rank 4 the adapter used 2,668 trainable parameters (0.47% of the model). At rank 16, 10,672 parameters (1.86%) matched full fine-tuning exactly on the held-out test set: 100%, the same score as training all 564,251 parameters.

The per-task view was the other lesson. The briefly-trained base model looked respectable overall at 83.25%, but it was weak exactly where a security tool cannot afford to be: MITRE mapping at 62.5% and log triage at 71.9%. The LoRA model brought every task to 100% on the test set. Aggregate accuracy hid the failures; the per-task breakdown exposed them.

Three working rules I took from this project:

1. Sweep on validation loss, not just accuracy — ties in accuracy are not ties in quality.
2. Always run the full fine-tuning baseline, or the parameter-efficiency claim means nothing.
3. Report per-task results. The average is where weak spots hide.

Scope, stated plainly: these numbers are on the project's synthetic instruction dataset, which is strongly separable — the experiment proves the fine-tuning pipeline and the efficiency effect, not production SOC performance. The same repo includes a PEFT/QLoRA script for fine-tuning Qwen2.5-0.5B on GPU; that run is documented as ready, and I have not claimed results for it because I have not run it yet.

Full code and a 39-page report (theory, implementation, sweep analysis, interview Q&A) are in my portfolio. If you work on applied LLM fine-tuning, I would value your critique.

#LLM #FineTuning #LoRA #QLoRA #CyberSecurity #PyTorch #MachineLearning #MLOps

---

## (c) DenseHAR Research Post

Fixed activity windows have a quiet failure mode, and my DenseHAR project was built to measure it.

Most human activity recognition systems classify one fixed window at a time: 128 sensor samples in, one activity label out. That works until a person changes activity mid-window — stands up, starts walking, sits back down. Then the window contains two activities and can only name one.

I built DenseHAR as the alternative: a network that predicts a label for every timestep (multi-scale 1D convolutions from my InceptionKAN work, SE attention, a dilated temporal stack, 68,898 parameters), evaluated on subjects the model had never seen — tuning and test subjects fully disjoint.

The headline comparison, on the project's synthetic IMU evaluation set:

• Overall accuracy: dense model 95.35%, windowed baseline 85.71%. A respectable gap, nothing dramatic.
• Inside transition windows only: dense model 90.63%, windowed baseline 48.03%.

The baseline loses 37.7 points at transitions; the dense model loses 4.7. That gap — invisible in any overall accuracy table — is the entire argument for dense segmentation, and it is why my earlier windowed work (InceptionKAN: 97.25% UCI-HAR, 98.45% PAMAP2, with ~238K parameters) is now being extended in this direction.

And the counter-result, which I am reporting because research posts should include them: on segmental F1 and edit score, the windowed baseline wins (F1@10: 0.899 vs 0.621). My dense model finds nearly every true boundary (recall 0.97) but also predicts too many (precision 0.36) — classic over-segmentation. The known remedy is MS-TCN-style refinement stages, which is the next experiment, together with the real-data run on transition-labelled datasets such as HAPT under leave-one-subject-out evaluation.

If you work on wearable sensing, time-series segmentation or on-device ML, I would be glad to compare notes. The code, evaluation protocol and a 39-page report are in my portfolio.

#HumanActivityRecognition #DeepLearning #TimeSeries #WearableComputing #PyTorch #Research #MachineLearning

---

## (d) DenseHAR on Real Data — The Synthetic Headline, the Tie, and the Fix

An honest follow-up to my DenseHAR post, because the first version of this story ended too early.

The headline I shared then — 95.35% accuracy, 90.63% inside transition windows — was measured on synthetic IMU data generated by my own simulator. A model can learn a generator's habits as easily as it learns human movement, so that number was always a pipeline proof, not a result. The test that counts is real data. Here it is.

The dataset is HAPT: real smartphone accelerometer and gyroscope recordings from 30 subjects, labelled per timestep with six activities and six postural transitions. My protocol was strict — subject-disjoint splits (train 21 subjects, validation 4, test 5 subjects the model never saw), normalisation statistics from train subjects only, and every design decision selected on validation before the test set was touched.

Phase 1 was a tie, and I am reporting it as one. DenseHAR exactly as originally built scored 78.86% per-timestep accuracy on real HAPT. The simple windowed baseline scored 78.91%. DenseHAR still won where it was designed to win — 59.55% vs 47.87% inside transition windows, 85.71% vs 77.30% on the transition-class timesteps — but it over-segmented real data badly (592 false boundaries against 50 true ones), and its segmental F1 collapsed. The synthetic design had not survived contact with real movement.

So phase 2 was a research exercise, not a tuning exercise. Every candidate lever was screened on validation only; only validation winners earned a single test evaluation. What the screening showed:

• Physics-respecting augmentation was the biggest lever (validation F1 0.9189 → 0.9584): rotating accelerometer and gyroscope jointly with one rotation matrix, magnitude warping, scaling, small time-warps. It attacks the actual problem — generalising to new people's movement.
• Giving the six transition classes their own labels during training (12-class supervision, mapped back to six classes for scoring) was the second lever.
• Post-processing — the unglamorous step — repaired the segmental collapse: on the phase-1 model it lifted validation segmental F1@25 from 0.641 to 0.866 with zero retraining.
• Two honest negatives: median-frequency class weights hurt (the worst validation run of the programme), and boundary-weighted cross-entropy — a centrepiece of the synthetic design — gave nothing over plain cross-entropy on real data.

The final system (augmentation + 12-class supervision + a 3-seed probability ensemble + validation-tuned post-processing), evaluated once on the five unseen test subjects:

• Per-timestep accuracy 90.38%, macro F1 0.9398 — vs 78.91% / 0.7258 for the windowed baseline under the identical protocol
• Transition-window accuracy 0.6093 vs 0.4787; accuracy on transition-class timesteps 87.44%
• Segmental F1@25 0.822 vs 0.749; edit score 84.88 vs 75.52
• Per class: walking and walking-upstairs F1 1.000 on unseen subjects, downstairs 0.994, laying 0.976 — and sitting (0.853) vs standing (0.815) remains the hard pair, as it is for every waist-worn HAR system

The synthetic 95.35% is still in the repo, labelled as what it is. The real headline is 90.38% on HAPT — a smaller number that means more, because every point of it survived data I did not generate. The full log of 22 experiments, including the failures, is in the project repo and the updated report.

If you work on wearable sensing or time-series segmentation, I would value your critique — especially on closing the sitting/standing gap.

#HumanActivityRecognition #DeepLearning #TimeSeries #WearableComputing #PyTorch #Research #MachineLearning

---

## (e) All Six Projects, Re-Run on Real Data — What Survived

An honest follow-up to the whole portfolio, because the first version of this story ended too early.

When I published these six projects, I stated the scope in the posts themselves: each result was measured on that project's own built-in evaluation data — in most cases, data generated by my own code. A model can learn a generator's habits as easily as it learns the task, so those headlines were always pipeline proofs, not results. The test that counts is data I did not generate. So I re-ran all six projects on public benchmarks, under one protocol throughout: baselines first, every improvement screened on validation or dev data only, and the final configuration evaluated on the test split once. Here is what survived, project by project.

1. DenseHAR — transition-aware activity recognition, on real HAPT sensor data with subjects the model never saw: 90.38% accuracy / 0.9398 macro F1, against 78.91% / 0.7258 for the windowed baseline under the identical protocol. The headline that held — smaller than the synthetic figure it replaced, and worth more for it.

2. Sentinel-LLM — cybersecurity classification on MITRE ATT&CK and on CVE severity data: the honest headline belongs to a classical model. A TF-IDF word 1–2-gram + linear SVM system scored 79.46% accuracy / 0.7788 macro F1 on ATT&CK — level on macro F1 with the published bar of 81.95% / 0.7795, under a different corpus and protocol — and 55.70% / 0.5444 on CVE severity, below the literature ceiling of ~70% set at 36k-CVE scale. My from-scratch LoRA model, the star of the synthetic run, scored 49.11% on ATT&CK and 48.90% on CVE: it lost to full fine-tuning and to TF-IDF. That is the LoRA lesson I will keep — a parameter-efficiency claim means nothing until the comparison survives real data. The packaged QLoRA script for a larger pretrained model is the documented next step; it has not been run, and I claim nothing for it.

3. AgentForge-RAG — retrieval on SciFact, via BEIR: the ladder climbed honestly — nDCG@10 0.5769 → 0.6269 → 0.6429 across naive, hybrid and reranked retrieval — and the final tuned pipeline reached nDCG@10 0.6466, Recall@5 0.7059, MRR@10 0.6090. It also stopped below the published BM25 bar of 0.665, and I am reporting that as what it is: a documented plateau for a purely lexical stack, not a win over it. RM3 pseudo-relevance feedback hurt and was rejected; the negative result is logged with the rest.

4. DiffusionForge — diffusion models from scratch, on MNIST downscaled to 16×16 so the existing U-Net applies unchanged (no FID claimed, and not comparable to full-resolution results): the one project where the real headline matched the synthetic one. DDPM sampling reached 100.0% class fidelity (70/70 samples) and DDIM-50 reached 84.29%, judged by a fidelity classifier that itself scores 98.13% on held-out real data — and a memorisation check came back clean, so those samples are novel generations, not copies of the training set.

5. FedGuard — federated learning with differential privacy, on MNIST with the full test set for every configuration: real data moved the numbers up, not down — centralised 0.9756, FedAvg IID 0.9620, FedAvg non-IID 0.9653 — which is its own warning that the synthetic table was measuring task difficulty, not method quality. FedProx, the hero of the synthetic run, did not recover the non-IID gap here (0.9520 — a negative result). Differentially private training held 0.9054 at epsilon-hat ≈ 5.55 and 0.9083 at epsilon-hat ≈ 10.2; epsilon-hat is a simplified estimate, and I label it as one every time, never as a formal guarantee.

6. ResolveDesk AI — support-ticket triage on the Bitext customer-support dataset, and the leakage catch of the programme. On a naive split the classifier reads 94.75% — until you notice that 13.79% of the naive test rows appear verbatim in the training rows. Deduplicate before splitting and the honest baseline is 89.9% / 0.8608. The final model, improved on validation only, reached 97.59% accuracy / 0.9748 macro F1 on intent and 99.5% / 0.995 on category under that deduplicated protocol. Priority prediction has no real labels in this data and remains synthetic-only; I am not presenting it as anything else.

Three lessons I am taking into every project from here:

• Synthetic numbers measure the generator, in both directions. Sentinel collapsed on real data; FedGuard rose. Neither direction was knowable from the synthetic tables.
• The leakage traps are real and material — temporal splitting for CVE data, duplicate utterances in Bitext. Both moved the headline by points, not decimals.
• Negative results are results. The LoRA model losing to TF-IDF, RM3 rejected, FedProx failing, a retrieval plateau below the published bar — every one is logged in the projects' results files, not deleted.

The synthetic runs are still in the repos, labelled as what they are. The real headline set is smaller, harder, and — for the first time — defensible line by line. Code, results files and reports for all six projects are in my portfolio. If you work on applied ML or ML evaluation, I would value your critique — especially on closing the gaps I have named rather than hidden.

#MachineLearning #DeepLearning #LLM #FineTuning #RAG #PyTorch #FederatedLearning #DiffusionModels #HumanActivityRecognition #AIEngineering

---

*Posting notes: attach the relevant "Results at a Glance" chart from the project's report
(`reports/assets/`) to posts (b) and (c) — Sentinel: `sentinel-tasks-params.png`;
DenseHAR: `densehar-transition-segmental.png`. For post (a), a simple collage of the six
report covers works well. Do not add metrics to the images that are not in the posts.*
