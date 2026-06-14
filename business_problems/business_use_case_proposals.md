# Deep Learning Final Project — Business Use Case Proposals

> Reference doc for the team. Based on the IE *Deep Learning — Final Project Guidelines* and the techniques covered across Blocks 1–4 of the course.

---

## 1. What the project requires (the hard constraints)

- **Predictive (non-generative) backend only** → must be **ANN, CNN, or RNN/LSTM/GRU**.
  - This **rules out** Block 3 material (GANs, DCGANs, autoencoders) and any LLM.
- **A real frontend** (Streamlit / v0 / Bolt encouraged) with **live, real-time** integration to the model.
- **A quantified business value proposition** (ROI / cost reduction / efficiency / revenue).
- **15-minute presentation**, live demo required, all members must speak.

### Grading weights
| Pillar | Weight |
|---|---|
| Technical depth & model architecture | 25% |
| MVP integration & frontend UX | 25% |
| Business use case & value proposition | 20% |
| Presentation & team delivery | 20% |
| Live demo & time management | 10% |

> **Takeaway:** Integration + live demo = **35%**. Favor an idea we can wire end-to-end *cleanly*, not the fanciest model.

---

## 2. Our toolkit (what the course already gave us working code for)

| Block | Architecture | Working notebooks we can reuse |
|---|---|---|
| 1 | **ANN** | Tabular regression (bike rentals), tabular classification (diabetes), multiclass (Fashion-MNIST), callbacks / early stopping |
| 2 | **CNN** | Image classification (Simpsons, cats/dogs, facial expression), **transfer learning + fine-tuning (ResNet)**, data augmentation |
| 4 | **RNN/LSTM** | Time-series forecasting (retail sales), sentiment analysis / NLP text classification |

These three lanes are the safest bets because we have reference code for each.

---

## 3. Proposals

### 🟢 ANN (tabular) — lowest risk, fastest to integrate

**1. Customer Churn Predictor (telecom / SaaS / subscription)**
- **What:** Input customer attributes → probability of churn + risk tier.
- **Value:** Retention is 5–7× cheaper than acquisition; flag the top X% at-risk for proactive offers.
- **Data:** Telco Customer Churn (Kaggle).
- **Reuse:** Mirrors the diabetes notebook. Frontend: form → risk gauge.

**2. Loan Default / Credit-Risk Scoring**
- **What:** Approve/decline + default probability.
- **Value:** Reduce loss rate, speed up decisioning.
- **Data:** Lending Club / German Credit.
- **Reuse:** Strong ROC-AUC story.

**3. Demand / Price Estimator** *(extends the bike-rental regression)*
- **What:** Predict daily demand from weather / season / calendar → staffing & inventory.
- **Value:** Cut stockouts and overstaffing.
- **Reuse:** We already have the dataset and model.

**4. Employee Attrition Predictor (HR)**
- **What:** Predict which employees are likely to leave.
- **Value:** People analytics; reduce costly turnover.
- **Data:** IBM HR Analytics. Same shape as churn.

---

### 🔵 CNN (vision) — highest "wow" for the live demo

**5. Automated Visual Quality Control / Defect Detection**
- **What:** Classify product images as defective / OK (castings, PCBs, surface cracks).
- **Value:** Replace manual inspection, catch defects earlier → scrap & warranty cost reduction.
- **Note:** Guidelines explicitly name "visual quality control." Transfer learning (ResNet) does the heavy lifting.

**6. Medical-Image Triage** (chest X-ray pneumonia, skin-lesion benign/malignant)
- **What:** Upload image → diagnosis + confidence.
- **Value:** Faster triage, fewer missed cases.
- **Data:** Kaggle chest X-ray / HAM10000.

**7. Retail Freshness Grading / Product Recognition** (fresh vs rotten produce)
- **What:** Classify product images for stock / quality checks.
- **Value:** Automate stock checks, reduce waste.
- **Reuse:** Uses the data-augmentation + fine-tuning notebooks directly.

---

### 🟣 RNN / LSTM (sequential) — best for forecasting or NLP

**8. Sales / Demand Forecasting Dashboard**
- **What:** Forecast next N periods of revenue or units.
- **Value:** Inventory planning, cash-flow.
- **Reuse:** This *is* the `RSCCASN` retail-sales notebook — minimal new code.

**9. Energy / Electricity Load Forecasting**
- **What:** Forecast power demand.
- **Value:** Grid balancing, procurement cost.
- **Reuse:** Same architecture as #8.

**10. Customer-Review Sentiment & Support-Ticket Router**
- **What:** Classify incoming text as positive/negative or by urgency/topic.
- **Value:** Prioritize angry customers, cut response time.
- **Reuse:** Directly reuses the sentiment-analysis notebook. Great live demo (type a review → instant sentiment).

---

## 4. Recommendation

Best grade-to-effort ratio — lead with one of these two:

- **Visual Quality Control (CNN #5)** — most impressive live demo (upload → instant verdict), explicitly blessed by the guidelines, and transfer learning means high accuracy with little training. *Pick if the team wants the "wow."*
- **Churn Predictor (ANN #1)** — cleanest end-to-end integration (a form wires trivially to Streamlit), strongest/clearest business ROI narrative, lowest technical risk. *Pick if the team wants a safe, polished MVP.*

**Hybrid option:** a "Customer Intelligence" app = churn ANN + a sentiment tab (RNN) on reviews — only if we have bandwidth. One solid model integrated flawlessly beats two half-wired ones (integration is 25%).

---

## 5. Quick comparison

| # | Idea | Arch | Code reuse | Demo impact | Risk |
|---|---|---|---|---|---|
| 1 | Customer churn | ANN | High | Medium | Low |
| 2 | Credit risk | ANN | High | Medium | Low |
| 3 | Demand estimator | ANN | Very High | Medium | Low |
| 4 | Employee attrition | ANN | High | Medium | Low |
| 5 | Visual QC / defects | CNN | High | **High** | Medium |
| 6 | Medical-image triage | CNN | High | **High** | Medium |
| 7 | Freshness grading | CNN | High | High | Medium |
| 8 | Sales forecasting | RNN | Very High | Medium | Low |
| 9 | Energy load forecast | RNN | High | Medium | Medium |
| 10 | Sentiment / ticket router | RNN | High | High | Low |
