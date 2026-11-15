# PneumoScan-X: Leak-Free Chest X-Ray Triage with Explainable Grad-CAM++

**Veritas AI International High School Hackathon 2026 | Track: AI + Health**

**Team:** Erlangga Azhar & Affan Alhafiz Hikren - *SMA Negeri 1 Kota Cilegon, Indonesia*

> **One-Sentence Pitch:** PneumoScan-X helps frontline clinicians in rural clinics without on-site radiologists prioritize high-risk chest X-rays for pneumonia, highlighting suspicious lung regions with Grad-CAM++ while eliminating dataset leakage.

* **Live Clinical Triage Web App:** `[Insert Your Streamlit Community Cloud URL Here]`

* **Full Research & Audit Notebook:** [`PneumoScan_X_Submission.ipynb`](./PneumoScan_X_Submission.ipynb) *(self-contained with embedded metrics, error audit, and Grad-CAM++ visualizations)*

## 1. Clinical Motivation

In primary care clinics across rural Indonesia (*Puskesmas*) and underserved regions worldwide, general practitioners often interpret chest radiographs without an on-site radiologist. Delayed recognition of bacterial or viral pneumonia leads to preventable morbidity, yet standard "black-box" deep learning classifiers fail in clinical practice due to three recurring flaws:

1. **Patient Data Leakage:** Naive random splits of public X-ray datasets place multiple scans from the same patient into both training and test sets, artificially inflating accuracy.

2. **Unsafe Default Thresholds:** Using an arbitrary $0.50$ probability cutoff misses subtle early infiltrates (high False Negatives).

3. **Lack of Trust & Input Guardrails:** Standard classifiers provide no visual justification and silently hallucinate medical predictions even when given non-X-ray color photos.

**PneumoScan-X** addresses all three barriers in a lightweight, CPU-friendly pipeline built around fine-tuned **MobileNetV2**.

## 2. Key Engineering Innovations

* **Strict Leak-Free Dataset Curation:** Automated **MD5 byte-hashing** removes nested folder duplicates and exact file copies in the Kaggle archive (reducing $11,712$ raw files down to $5,824$ **unique radiographs** across $2,789$ **unique patients**). `GroupShuffleSplit` grouped by `patient_id` ensures **zero patient overlap** across Train ($3,969$), Validation ($943$), and Test ($912$) splits.

* **Pulmonary Preprocessing & Out-of-Distribution (OOD) Guard:** Applies an RGB channel-variance check (`validate_xray=True`) to reject accidental color photo uploads before inference, followed by a $5\%$ border crop and **CLAHE** ($\text{clipLimit} = 1.5$) to standardize lung contrast.

* $F_2$**-Optimal Clinical Threshold Calibration (**$0.12$**):** Calibrated strictly on the validation set to maximize the $F_2$-score (weighting Recall $2\times$ higher than Precision), mapping probabilities into a **3-Tier Clinical Triage** workflow:

  * **Normal / Low Risk (**$p < 0.12$**):** Routine clinical workflow.

  * **Suspect / Priority Review (**$0.12 \le p < 0.50$**):** Flagged for secondary physician review.

  * **High-Probability Pneumonia (**$p \ge 0.50$**):** Immediate clinical escalation.

* **Explainable Grad-CAM++ Localization:** Generates class-discriminative heatmaps from the final convolutional block (`features[-1]`) so clinicians can verify pathological consolidation rather than relying on a raw score.

## 3. Benchmark Results (Unseen Test Patients, $N = 912$)

Evaluated on $912$ radiographs from patients never seen during training or threshold calibration ($652$ Pneumonia, $260$ Normal):

| Method | Accuracy | Recall (Sensitivity) | Precision | $F_2$-Score | AUC-ROC | Missed Cases (FN) | False Alarms (FP) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| Trivial Baseline (Always Pneumonia) | $71.49\%$ | $100.00\%$ | $71.49\%$ | $0.9261$ | $0.5000$ | $0 / 652$ ($0.00\%$) | $260 / 260$ ($100.0\%$) | 
| MobileNetV2 (Default Cutoff $= 0.50$) | $96.49\%$ | $97.24\%$ | $97.84\%$ | $0.9736$ | $0.9885$ | $18 / 652$ ($2.76\%$) | $14 / 260$ ($5.38\%$) | 
| **PneumoScan-X (Final Cutoff** $= 0.12$**)** | $92.54\%$ | $98.93\%$ | $91.36\%$ | $0.9731$ | $0.9885$ | $7 / 652$ **(**$1.07\%$**)** | $61 / 260$ **(**$-76.5\%$**)** | 

* **High Sensitivity:** Catches $645 / 652$ ($98.93\%$) of pneumonia cases on unseen patients, cutting missed diagnoses by $61.1\%$ compared to the standard $0.50$ cutoff ($18 \rightarrow 7$ False Negatives).

* **Triage Efficiency:** Reduces false alarms by $76.5\%$ compared to a naive all-positive screening rule while maintaining fast CPU inference.

## 4. Repository Structure

```
PneumoScan-X/
├── README.md                        # Project overview, clinical benchmarks, & setup guide
├── PneumoScan_X_Submission.ipynb    # 11-part research narrative notebook with embedded outputs
├── pneumoscan_engine.py             # Core engine: MD5 deduplication, CLAHE, OOD guard, & Grad-CAM++
├── app.py                           # Streamlit clinical triage web application
├── best_mobilenet_pneumonia.pth     # Fine-tuned MobileNetV2 model weights
├── requirements.txt                 # Python dependencies for local & Streamlit Cloud deployment
├── .gitignore                       # Excludes raw dataset archives and local caches
└── sample_images/                   # Sample radiographs & OOD color image for instant testing

```

## 5. Quickstart: Run Locally or Reproduce Research

### Option A: Launch the Clinical Triage Web App (`app.py`)

```
git clone https://github.com/erlangga-azhar/PneumoScan-X.git
cd PneumoScan-X
pip install -r requirements.txt
streamlit run app.py

```

Upload any chest X-ray (or select one of the built-in samples in `sample_images/`) to inspect the 3-tier triage recommendation and Grad-CAM++ lung overlay.

### Option B: Run the Research & Audit Notebook (`PneumoScan_X_Submission.ipynb`)

1. Open [`PneumoScan_X_Submission.ipynb`](./PneumoScan_X_Submission.ipynb) directly on GitHub to view all pre-computed tables, confusion matrices, and Grad-CAM++ error audit grids.

2. To re-execute from scratch in **Google Colab**, upload the `.ipynb` file and click **Runtime** $\rightarrow$ **Run all**. Cell 1 automatically fetches `pneumoscan_engine.py`, `best_mobilenet_pneumonia.pth`, and the public Kaggle dataset via `kagglehub` with zero manual uploads or API keys.

## 6. Forensic Error Audit & Limitations

* **External Medical Artifacts:** Auditing the $7$ false negatives on the test set revealed that external hardware such as circular ECG electrode pads and portable `"A-P"` markers (e.g., `person1453_bacteria_3770.jpeg`) can occasionally distract convolutional activations outside the lung fields. Future iterations will incorporate a **U-Net lung segmentation mask** prior to classification.

* **Retrospective Pediatric Cohort:** The underlying dataset consists primarily of retrospective pediatric radiographs from Guangzhou Women and Children's Medical Center. Clinical adoption in Indonesian primary care clinics requires prospective multi-center validation across adult and geriatric populations.

* **Medical Disclaimer:** *PneumoScan-X is an educational and research prototype developed for the Veritas AI Hackathon 2026. It is not a certified medical device and must not be used as a substitute for professional radiological diagnosis.*

## 7. Sources & Acknowledgments

* **Dataset:** Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification* (hosted on Kaggle by Paul Mooney).

* **Frameworks & Libraries:** PyTorch, Torchvision (`MobileNetV2`), OpenCV (`opencv-python-headless`), `pytorch-grad-cam`, Scikit-learn, Pandas, NumPy, Streamlit.

* **AI Tools Disclosure:** AI coding assistants were used to help debug OpenCV/Grad-CAM++ tensor shape edge cases and structure the Streamlit interface; all dataset deduplication logic, patient-aware splits, and clinical evaluation metrics were independently designed and verified by the team.