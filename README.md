# 🫁 PneumoScan-X

### Leak-Free Chest X-Ray Pneumonia Triage with Explainable Grad-CAM++

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erlangga-azhar/PneumoScan-X/blob/main/PneumoScan_X_Submission_v4.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pneumoscan-x.streamlit.app/)
[![Veritas AI 2026](https://img.shields.io/badge/Veritas%20AI%20Hackathon%202026-AI%20%2B%20Health-00A86B?style=flat-square)](https://github.com/erlangga-azhar/PneumoScan-X)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CLAHE-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F2%20Calibration-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Bilingual%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

![Unseen ROC-AUC](https://img.shields.io/badge/Unseen%20ROC--AUC-0.9885-brightgreen?style=flat-square)
![Sensitivity](https://img.shields.io/badge/Sensitivity%20(Recall)-98.93%25-brightgreen?style=flat-square)
![NPV](https://img.shields.io/badge/Negative%20Predictive%20Value-96.60%25-brightgreen?style=flat-square)
![CPU Latency](https://img.shields.io/badge/CPU%20Forward%20Latency-26.46%20ms-informational?style=flat-square)
![Leakage Check](https://img.shields.io/badge/Patient%20Overlap-0%20Patients%20(100%25%20Leak--Free)-blueviolet?style=flat-square)

---

**Authors:** **Erlangga Azhar** & **Affan Alhafiz Hikren** *(SMA Negeri 1 Kota Cilegon, Banten, Indonesia)*

[🚀 Live Web App](https://pneumoscan-x.streamlit.app/) · [📓 Reproducible Notebook](PneumoScan_X_Submission_v4.ipynb) · [📊 Benchmark Results](#-4-benchmark-results-unseen-test-set-n--912) · [🔍 Forensic Error Audit](#-6-forensic-error-audit-limitations--failure-modes) · [🗺️ Engineering Roadmap](#-7-limitations--engineering-roadmap)

**PneumoScan-X** is an end-to-end, leak-free deep learning clinical triage pipeline built for frontline healthcare workers in rural clinics without on-site radiologists. It prioritizes suspected pneumonia chest radiographs in **$\approx 26.46\text{ ms}$ on standard CPUs**, pairs risk scores with **Grad-CAM++** visual explanations, blocks accidental non-X-ray photo uploads via an **RGB channel-variance Out-of-Distribution (OOD) guard**, and cuts missed pneumonia diagnoses by **$65.0\%$** through validation-calibrated $F_2$-score threshold optimization.

> [!CAUTION]
> **Medical Disclaimer:** *PneumoScan-X is a research and educational decision-support prototype developed for the Veritas AI International High School Hackathon 2026. It has not undergone regulatory clinical validation (Kemenkes RI / FDA / CE) and must not be used for primary clinical diagnosis.*

---

## 🎯 1. Clinical Motivation & Problem Statement

In underserved primary healthcare facilities (such as rural *Puskesmas* across Indonesia), chest radiographs are routinely acquired without immediate radiologist coverage. Standard medical image classifiers typically optimize overall accuracy at a default `0.50` probability cutoff, creating three critical deployment risks:

1. **High-Cost False Negatives:** Missing an active pneumonia case delays life-saving antibiotic or antiviral therapy.
2. **Inflated Metrics from Dataset Leakage:** Public chest X-ray archives contain nested duplicate folders and multiple follow-up scans per patient. Naive random splits leak patient anatomy across training and test sets, artificially inflating benchmark scores.
3. **Silent Failure on Invalid Inputs:** Standard vision pipelines silently convert accidental smartphone camera snaps or color images into grayscale and output confident false medical alerts.

PneumoScan-X solves all three challenges through **MD5 + patient-grouped splitting**, **validation-calibrated $F_2$ triage**, **OOD input guarding**, and **Grad-CAM++ visual interpretability**.

---

## 🏗️ 2. System Architecture & Triage Pipeline

```text
┌─────────────────────┐    ┌────────────────────────────────┐    ┌──────────────────────────────────┐
│  Raw Radiograph     │───▶│  RGB Channel-Variance OOD      │───▶│  5% Safe Border Crop + CLAHE     │
│  Input (.jpg/.png)  │    │  Guard (Blocks Color Photos)   │    │  (clipLimit=1.5, 8x8 Grid, 224²) │
└─────────────────────┘    └────────────────────────────────┘    └─────────────────┬────────────────┘
                                                                                   │
┌─────────────────────┐    ┌────────────────────────────────┐    ┌─────────────────▼────────────────┐
│  Grad-CAM++ Heatmap │◀───│  Validation-Calibrated F2      │◀───│  Fine-Tuned MobileNetV2 Backbone │
│  (features[-1],     │    │  3-Tier Triage (Cutoff = 0.12) │    │  (features[14:] Unfrozen, CPU)   │
│   cutoff = 0.20)    │    │  Low Risk | Suspect | High     │    │  ~26.46 ms Forward Latency       │
└─────────────────────┘    └────────────────────────────────┘    └──────────────────────────────────┘
```

### ✨ Key Engineering Contributions

* **Strict MD5 Deduplication ($11,712 \rightarrow 5,824$ images):** Eliminated nested directory copies (`chest_xray/chest_xray/`) and byte-identical cross-split duplicates via MD5 byte-hashing.
* **Zero Patient Leakage ($2,789$ unique patients):** Normalized patient IDs across all hospital naming conventions (`personXXXX`, `IM-XXXX`, `NORMAL2-IM-XXXX`) and enforced disjoint `GroupShuffleSplit` partitions ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
* **Artifact-Mitigating Preprocessing:** Applied a **5% safe border crop** (removing bright scan edges and portable `"A-P"` markers) followed by **CLAHE** (`clipLimit=1.5`, $8 \times 8$ tile grid) to standardize lung field contrast.
* **CPU-Ready MobileNetV2 Backbone:** Fine-tuned upper convolutional blocks (`features[14:]`) with differential AdamW learning rates ($10^{-5}$ backbone / $10^{-4}$ classifier head) and class-weighted Cross-Entropy loss, achieving **$26.46\text{ ms}$** mean forward-pass latency on standard CPUs.
* **Post-Hoc Validation-Calibrated $F_2$ Cutoff ($0.50 \rightarrow 0.12$):** Tuned exclusively on the validation split to weight Recall $2\times$ higher than Precision, reducing missed pneumonia radiographs on the unseen test set by **$65.0\%$** ($20 \rightarrow 7$ out of $652$) while boosting **Negative Predictive Value (NPV) to $96.60\%$**.
* **OOD Safety Guard & Grad-CAM++ Explainability:** Rejects non-grayscale uploads prior to inference and overlays pulmonary saliency maps (activation threshold `0.20`) alongside a 3-tier clinical worklist.

---

## 🧬 3. Dataset & Leak-Free Split Architecture

**Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kermany et al. / Paul Mooney, Kaggle).

| Partition | Unique Radiographs | Unique Patients | Pneumonia Cases | Normal Controls | Patient Overlap with Other Splits |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 3,969 | 1,952 | 2,911 (73.3%) | 1,058 (26.7%) | **0 patients (0.0%)** |
| **Validation** | 943 | 418 | 682 (72.3%) | 261 (27.7%) | **0 patients (0.0%)** |
| **Test (Unseen)** | 912 | 419 | 652 (71.5%) | 260 (28.5%) | **0 patients (0.0%)** |
| **Total Verified** | **5,824** | **2,789** | **4,245 (72.9%)** | **1,579 (27.1%)** | **100% Leak-Free** |

---

## 📊 4. Benchmark Results (Unseen Test Set, $N = 912$)

> [!IMPORTANT]
> To avoid conflating network training with post-hoc decision calibration, we report results across three progressive evaluation stages on the unseen patient-disjoint test set ($N = 912$ radiographs across $419$ unique patients).

| Evaluation Metric | Trivial Baseline (Always Pneumonia) | Stage 1: MobileNetV2 (Cutoff $= 0.50$) | Stage 2: PneumoScan-X ($F_2$ Cutoff $= 0.12$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 71.49% | **95.72%** | 92.54% |
| **Sensitivity (Recall)** | 100.00% | 96.93% | **98.93%** |
| **Specificity** | 0.00% | **92.69%** | 76.54% |
| **Precision (PPV)** | 71.49% | **97.08%** | 91.36% |
| **Negative Predictive Value (NPV)** | — | 92.34% | **96.60%** |
| **$F_2$-Score** | 0.9261 | 0.9696 | **0.9731** |
| **ROC-AUC** | 0.5000 | **0.9885** | **0.9885** |
| **Missed Cases (FN)** | 0 / 652 (0.00%) | 20 / 652 (3.07%) | **7 / 652 (1.07% · -65.0%)** |
| **False Alarms (FP Rate)** | 260 / 260 (100.0%) | 19 / 260 (7.31%) | **61 / 260 (23.46% · -76.5% vs. Baseline)** |

### ⚡ Inference Latency (Single Radiograph, CPU)

* **Mean Forward-Pass Latency:** `26.46 ms`
* **P95 Forward-Pass Latency:** `32.84 ms`

---

## 🚦 5. Three-Tier Clinical Triage & OOD Safety System

Rather than exposing raw logits to busy frontline clinicians, PneumoScan-X maps calibrated risk scores into three actionable triage queues:

| Triage Tier | Probability Range | Clinical Action Protocol |
| :--- | :---: | :--- |
| 🟢 **NORMAL / LOW RISK** | $p < 0.12$ | No significant consolidation pattern detected ($96.60\%$ NPV). Proceed with routine clinical follow-up. |
| 🟡 **SUSPECT / PRIORITY REVIEW** | $0.12 \le p < 0.50$ | Early or borderline opacity above the $F_2$ cutoff (`0.12`). Prioritize secondary physician or teleradiology review. |
| 🔴 **HIGH PROBABILITY PNEUMONIA** | $p \ge 0.50$ | Strong pulmonary consolidation signature. Expedite immediate clinical intervention and isolation protocols. |

---

## 🔍 6. Forensic Error Audit (Limitations & Failure Modes)

To ensure clinical transparency, we audited all **7 false negatives** out of $652$ positive test radiographs at the `0.12` triage cutoff using Grad-CAM++:

1. **External Medical Hardware Distractions:** On `person1453_bacteria_3770.jpeg` ($p = 0.0657$), Grad-CAM++ saliency shifted away from lung parenchyma onto circular ECG electrode leads and portable `"A-P"` markers.
2. **Subtle Early Viral Infiltrates:** On `person1461_virus_2510.jpeg` ($p = 0.0439$), Grad-CAM++ localized bilateral lower-lobe opacity accurately, but faint radiological contrast kept the predicted score below `0.12`.

---

## 🗺️ 7. Limitations & Engineering Roadmap

1. **External Medical Artifacts & Monochromatic OOD Blind Spot:** Our lightweight RGB channel-variance check blocks color photos (`ValueError`), but cannot reject monochromatic non-chest images (e.g., hand radiographs) or mask out ECG electrodes.
   * *Planned Fix:* Prepend a **lightweight U-Net lung-field segmentation mask** and anatomical mask-area gate so the classifier inspects only verified lung parenchyma.
2. **Uncalibrated Softmax Probabilities:** Training with class-weighted Cross-Entropy and evaluating at an $F_2$-tuned cutoff produces relative risk scores rather than calibrated epidemiological probabilities.
   * *Planned Fix:* Apply **Temperature Scaling / Platt Calibration** on validation logits and track **Expected Calibration Error (ECE)** and **Brier Score**.
3. **Binary Formulation Without Differential Diagnosis:** Current predictions separate `NORMAL` vs. `PNEUMONIA` without distinguishing bacterial vs. viral etiology or screening for high-burden differentials in Indonesian primary care such as **Tuberculosis (TB)**, pleural effusion, or cardiomegaly.
   * *Planned Fix:* Expand to multi-label thoracic pathology detection using NIH ChestX-ray14, CheXpert, and regional TB cohorts.
4. **Single-Center Pediatric Cohort & Projection Shift:** The underlying Kaggle dataset comprises retrospective pediatric scans (ages 1–5) from a single center, whereas *Puskesmas* clinics serve adult and geriatric populations across both upright PA and portable AP views.
   * *Planned Fix:* Perform multi-center external validation across adult/geriatric cohorts with subgroup sensitivity reporting.
5. **Training History Serialization:** Current checkpoints store weights and `optimal_threshold` without persisting per-epoch loss and $F_2$ curves.
   * *Planned Fix:* Embed full per-epoch training logs inside `.pth` metadata and plot convergence curves directly in the notebook.
6. **Prospective Clinical Reader Study:** Offline retrospective metrics must be validated in live clinical workflows prior to regulatory submission (Kemenkes RI / FDA / CE).
   * *Planned Fix:* Conduct a prospective human-AI reader study with general practitioners measuring triage turnaround time and diagnostic sensitivity with vs. without Grad-CAM++ overlays.

---

## 📁 8. Repository Structure

```text
PneumoScan-X/
├── PneumoScan_X_Submission_v4.ipynb   # End-to-end reproducible Google Colab notebook (MD5, splits, metrics, Grad-CAM++)
├── app.py                             # Bilingual (EN/ID) Streamlit clinical triage & Grad-CAM++ web application
├── pneumoscan_engine.py               # Deduplication, patient-split, CLAHE preprocessing, model & evaluation utilities
├── best_mobilenet_pneumonia.pth       # Fine-tuned MobileNetV2 weights + validation-calibrated threshold (0.12)
└── README.md                          # Project documentation, benchmarks, error audit & roadmap
```

---

## 🚀 9. Quickstart

### Option A: Live Clinical Triage Web App
Launch the interactive bilingual (**English / Bahasa Indonesia**) triage interface directly in your browser:

👉 **[https://pneumoscan-x.streamlit.app/](https://pneumoscan-x.streamlit.app/)**

### Option B: One-Click Google Colab Notebook
1. Open [`PneumoScan_X_Submission_v4.ipynb`](https://colab.research.google.com/github/erlangga-azhar/PneumoScan-X/blob/main/PneumoScan_X_Submission_v4.ipynb) in Google Colab.
2. Click **Runtime $\rightarrow$ Restart session and run all**.
3. The notebook automatically installs dependencies, fetches `pneumoscan_engine.py` and `best_mobilenet_pneumonia.pth`, downloads the Kaggle dataset via `kagglehub`, and reproduces every table and Grad-CAM++ figure.

### Option C: Local Python Setup & Inference

```bash
git clone https://github.com/erlangga-azhar/PneumoScan-X.git
cd PneumoScan-X
pip install torch torchvision scikit-learn opencv-python-headless matplotlib pandas numpy grad-cam kagglehub streamlit
```

**Run single-radiograph triage in Python:**

```python
import torch
from torchvision import transforms
from pneumoscan_engine import load_trained_model, preprocess_xray_cv, format_clinical_triage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, opt_thresh = load_trained_model("best_mobilenet_pneumonia.pth", device)

tf_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Applies RGB variance OOD check, 5% border crop, and CLAHE contrast standardization
img_np = preprocess_xray_cv("sample_xray.jpeg", validate_xray=True)
tensor = tf_eval(img_np).unsqueeze(0).to(device)

with torch.no_grad():
    prob = torch.softmax(model(tensor), dim=1)[0, 1].item()

tier, _, tier_margin = format_clinical_triage(prob, threshold=opt_thresh)
print(f"Pneumonia Probability: {prob:.2%} | Triage Tier: {tier} (Margin: {tier_margin:.1f}%)")
```

**Launch the Streamlit web app locally:**

```bash
streamlit run app.py
```

---

## 📚 10. Sources & Team

* **Dataset:** Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification* (hosted on Kaggle by Paul Mooney).
* **Frameworks & Libraries:** PyTorch, Torchvision (`MobileNetV2`), OpenCV (`opencv-python-headless`), `pytorch-grad-cam`, Scikit-learn, Pandas, NumPy, Streamlit.
* **Team:** **Erlangga Azhar** & **Affan Alhafiz Hikren** — SMA Negeri 1 Kota Cilegon, Banten, Indonesia.