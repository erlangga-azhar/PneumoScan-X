import hashlib
import io
import os
import re
import shutil
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pneumoscan_engine import (
    preprocess_xray_cv,
    load_trained_model,
    format_clinical_triage,
)

# ==========================================
# PAGE CONFIG & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="PneumoScan-X | Clinical Chest X-Ray Triage",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

WEIGHTS_PATH = "best_mobilenet_pneumonia.pth"
GITHUB_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/erlangga-azhar/PneumoScan-X/main/"
    "best_mobilenet_pneumonia.pth"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYNTHETIC_OOD_TOKEN = "__SYNTHETIC_OOD_TEST__"

TF_EVAL = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# BILINGUAL UI DICTIONARY (EN / ID)
# ==========================================
I18N = {
    "EN": {
        "subtitle": "Veritas AI Hackathon 2026 · AI + Health",
        "team_label": "Team",
        "school_label": "School",
        "guide_btn": "📖 Quick Clinical & Demo Guide",
        "guide_title": "PneumoScan-X · Quick Guide",
        "guide_body": """
**1. Input Radiograph or Synthetic OOD Test**
- Upload an AP/PA chest X-ray (`JPEG`/`PNG`) or pick a built-in test case.
- Select **`[Test OOD Guard] Synthetic Non-X-Ray Color Image`** to verify how the RGB channel guard blocks non-radiograph photos in one click.

**2. F2-Calibrated 3-Zone Triage (`{t_low}` Cutoff)**
- **Normal (`< {t_low}`):** Low risk; below the F2 sensitivity threshold.
- **Suspect / Priority (`{t_low} – {t_high_prev}`):** Exceeds the `{t_low}` F2 cutoff (`98.93%` recall); flag for secondary clinical review.
- **High Risk (`>= {t_high}`):** Strong pneumonia consolidation signature; prioritize immediate medical evaluation.

**3. Grad-CAM++ Localization & Cutoff**
- Use the **Activation Cutoff** slider to filter out low-level background noise in real time.
- Color intensity reflects **relative gradient attention** (Cyan/Green $\\rightarrow$ Yellow/Red = peak model focus), not direct infection severity.

**4. Export Clinical Triage Artifacts**
- Download the **Side-by-Side Composite PNG** and **Triage Note TXT** below the scan results for referral documentation.
""",
        "calib_header": "Triage Calibration",
        "cam_slider": "Grad-CAM++ Activation Cutoff",
        "cam_help": "Suppresses low-intensity background activations below this threshold.",
        "ood_checkbox": "Enforce RGB OOD Guard",
        "ood_help": "Rejects non-grayscale color photographs before clinical inference.",
        "perf_header": "Validated Performance (N = 912)",
        "perf_bullets": (
            "- **Sensitivity (Recall):** `98.93%` (`645 / 652`)\n"
            "- **AUC-ROC:** `0.9885`\n"
            "- **F2-Score:** `0.9731`\n"
            "- **F2 Triage Cutoff:** `{t_low}`\n"
            "- **False Alarm Reduction:** `-76.5%` vs. baseline"
        ),
        "disclaimer": (
            "**Prototype Disclaimer:** PneumoScan-X is a research prototype for clinical triage "
            "prioritization and is **not** a substitute for certified radiological diagnosis."
        ),
        "main_title": "PneumoScan-X: Chest X-Ray Pneumonia Triage",
        "main_desc": (
            "Frontline decision-support prototype combining **leak-free MobileNetV2 inference**, "
            "**Out-of-Distribution (OOD) input filtering**, **F2-calibrated 3-tier triage**, and "
            "**Grad-CAM++** pulmonary localization."
        ),
        "source_radio": "Select Radiograph Input Source:",
        "opt_upload": "Upload Chest X-Ray Image",
        "opt_preset": "Use Built-in Sample Preset",
        "uploader_label": "Upload an anterior-posterior / posterior-anterior chest radiograph (JPEG/PNG)",
        "preset_select": "Choose a test case:",
        "synthetic_ood_label": "[Test OOD Guard] Synthetic Non-X-Ray Color Image",
        "decode_err": "Could not decode image `{path}`.",
        "bar_title": "3-Zone F2 Triage Position",
        "bar_scale": "{zero}  |  F2: {t_low}  |  High: {t_high}  |  {one}",
        "bar_zone_normal": "Normal (< {t_low})",
        "bar_zone_suspect": "Suspect ({t_low}–{t_high_prev})",
        "bar_zone_high": "High Risk (>= {t_high})",
        "metric_prob": "Pneumonia Probability",
        "metric_cutoff": "Triage Cutoff (F2)",
        "metric_zone": "Triage Zone",
        "metric_conf": "Zone Confidence",
        "col1_title": "1. Standardized Radiograph",
        "col1_cap": "{name} · 5% Border Crop + CLAHE (clipLimit=1.5)",
        "col2_title": "2. Grad-CAM++ Pulmonary Focus",
        "col2_cap": "Grad-CAM++ Overlay (Activation Cutoff >= {cutoff})",
        "cam_legend": (
            "**Grad-CAM++ Color Scale:** Unshaded = Below cutoff (`< {cutoff}`) · "
            "Cyan/Green = Moderate activation · Yellow/Red = Peak model attention "
            "*(relative gradient focus, not direct pathology severity)*."
        ),
        "normal_note": (
            "Note: Because this radiograph falls below the {t_low} clinical triage cutoff, "
            "Grad-CAM++ highlights weak sub-threshold features rather than confirmed pathology."
        ),
        "export_header": "Export Clinical Triage Artifacts",
        "dl_img_btn": "⬇️ Download Composite Image (.png)",
        "dl_txt_btn": "⬇️ Download Triage Note (.txt)",
        "ood_err_title": "Out-of-Distribution (OOD) Input Rejected",
        "ood_err_body": (
            "PneumoScan-X detected non-grayscale color channel variance. Please upload a valid "
            "grayscale chest X-ray radiograph."
        ),
        "runtime_err": "Unexpected inference error on `{name}`: `{err}`",
        "load_err": (
            "Failed to load model weights (`{path}`). Ensure the `.pth` file exists in the "
            "repository root or is accessible on GitHub. Details: `{err}`"
        ),
    },
    "ID": {
        "subtitle": "Veritas AI Hackathon 2026 · AI + Kesehatan",
        "team_label": "Tim",
        "school_label": "Sekolah",
        "guide_btn": "📖 Panduan Singkat Klinis & Demo",
        "guide_title": "PneumoScan-X · Panduan Singkat",
        "guide_body": """
**1. Input Radiograf atau Uji Sintetis OOD**
- Unggah foto rontgen dada AP/PA (`JPEG`/`PNG`) atau pilih kasus uji bawaan.
- Pilih **`[Uji Guard OOD] Gambar Sintetis Berwarna Non-X-Ray`** untuk menguji pemblokiran foto non-rontgen dalam satu klik.

**2. Triase 3-Zona Terkalibrasi F2 (Ambang `{t_low}`)**
- **Normal (`< {t_low}`):** Risiko rendah; berada di bawah ambang sensitivitas F2.
- **Suspek / Prioritas (`{t_low} – {t_high_prev}`):** Melampaui ambang F2 `{t_low}` (`98,93%` recall); tandai untuk tinjauan klinis lanjutan.
- **Risiko Tinggi (`>= {t_high}`):** Indikasi kuat konsolidasi pneumonia; prioritaskan evaluasi dokter segera.

**3. Lokalisasi Grad-CAM++ & Ambang Aktivasi**
- Geser **Ambang Aktivasi Grad-CAM++** untuk menyaring derau latar belakang intensitas rendah secara instan.
- Skala warna menunjukkan **fokus perhatian gradien relatif** (Biru Muda/Hijau $\\rightarrow$ Kuning/Merah = fokus puncak model), bukan tingkat keparahan infeksi langsung.

**4. Ekspor Artefak Triase Klinis**
- Unduh **Gambar Gabungan PNG** dan **Catatan Triase TXT** di bawah hasil pemindaian untuk dokumentasi rujukan.
""",
        "calib_header": "Kalibrasi Triase",
        "cam_slider": "Ambang Aktivasi Grad-CAM++",
        "cam_help": "Menekan aktivasi latar belakang berintensitas rendah di bawah nilai ambang ini.",
        "ood_checkbox": "Aktifkan Guard OOD RGB",
        "ood_help": "Menolak foto berwarna non-grayscale sebelum inferensi klinis dijalankan.",
        "perf_header": "Performa Tervalidasi (N = 912)",
        "perf_bullets": (
            "- **Sensitivitas (Recall):** `98,93%` (`645 / 652`)\n"
            "- **AUC-ROC:** `0,9885`\n"
            "- **Skor F2:** `0,9731`\n"
            "- **Ambang Triase F2:** `{t_low}`\n"
            "- **Penurunan Alarm Palsu:** `-76,5%` dibanding baseline"
        ),
        "disclaimer": (
            "**Penafian Prototipe:** PneumoScan-X adalah prototipe riset untuk membantu prioritas "
            "triase klinis dan **bukan** pengganti diagnosis dokter spesialis radiologi."
        ),
        "main_title": "PneumoScan-X: Triase Pneumonia Rontgen Dada",
        "main_desc": (
            "Prototipe pendukung keputusan klinis lini depan yang menggabungkan **inferensi MobileNetV2 bebas kebocoran data**, "
            "**penyaringan input Out-of-Distribution (OOD)**, **triase 3-zona terkalibrasi F2**, serta "
            "**lokalisasi paru Grad-CAM++**."
        ),
        "source_radio": "Pilih Sumber Input Radiograf:",
        "opt_upload": "Unggah Foto Rontgen Dada",
        "opt_preset": "Gunakan Sampel Bawaan",
        "uploader_label": "Unggah foto rontgen dada proyeksi AP / PA (JPEG/PNG)",
        "preset_select": "Pilih kasus uji:",
        "synthetic_ood_label": "[Uji Guard OOD] Gambar Sintetis Berwarna Non-X-Ray",
        "decode_err": "Gagal membaca file gambar `{path}`.",
        "bar_title": "Posisi Probabilitas pada Triase 3-Zona F2",
        "bar_scale": "{zero}  |  F2: {t_low}  |  Tinggi: {t_high}  |  {one}",
        "bar_zone_normal": "Normal (< {t_low})",
        "bar_zone_suspect": "Suspek ({t_low}–{t_high_prev})",
        "bar_zone_high": "Risiko Tinggi (>= {t_high})",
        "metric_prob": "Probabilitas Pneumonia",
        "metric_cutoff": "Ambang Triase (F2)",
        "metric_zone": "Zona Triase",
        "metric_conf": "Keyakinan Zona",
        "col1_title": "1. Radiograf Terstandarisasi",
        "col1_cap": "{name} · Pemotongan Tepi 5% + CLAHE (clipLimit=1,5)",
        "col2_title": "2. Fokus Paru Grad-CAM++",
        "col2_cap": "Overlay Grad-CAM++ (Ambang Aktivasi >= {cutoff})",
        "cam_legend": (
            "**Skala Warna Grad-CAM++:** Tanpa warna = Di bawah ambang (`< {cutoff}`) · "
            "Biru Muda/Hijau = Aktivasi sedang · Kuning/Merah = Fokus perhatian puncak model "
            "*(intensitas gradien relatif, bukan ukuran keparahan patologi secara langsung)*."
        ),
        "normal_note": (
            "Catatan: Karena radiograf ini berada di bawah ambang triase klinis {t_low}, "
            "aktivasi Grad-CAM++ merepresentasikan fitur lemah di bawah ambang batas, bukan patologi terkonfirmasi."
        ),
        "export_header": "Ekspor Artefak Triase Klinis",
        "dl_img_btn": "⬇️ Unduh Gambar Gabungan (.png)",
        "dl_txt_btn": "⬇️ Unduh Catatan Triase (.txt)",
        "ood_err_title": "Input Out-of-Distribution (OOD) Ditolak",
        "ood_err_body": (
            "PneumoScan-X mendeteksi variansi kanal warna non-grayscale. Harap unggah citra "
            "rontgen dada grayscale yang valid."
        ),
        "runtime_err": "Terjadi kesalahan inferensi pada `{name}`: `{err}`",
        "load_err": (
            "Gagal memuat bobot model (`{path}`). Pastikan file `.pth` berada di direktori utama "
            "atau dapat diakses di GitHub. Detail: `{err}`"
        ),
    },
}


def fmt_num(val: float, lang: str, decimals: int = 2) -> str:
    """Formats decimals consistently using dot for EN and comma for ID."""
    s = f"{val:.{decimals}f}"
    return s.replace(".", ",") if lang == "ID" else s


def fmt_pct(val: float, lang: str, decimals: int = 2) -> str:
    """Formats percentages consistently using dot for EN and comma for ID."""
    s = f"{val:.{decimals}%}"
    return s.replace(".", ",") if lang == "ID" else s


def localize_triage_output(tier: str, desc: str, lang: str, opt_thresh: float):
    """Localizes clinical triage banner and zone labels dynamically using opt_thresh."""
    if lang == "EN":
        return tier, desc, tier.split("/")[0].strip()

    thresh_id = fmt_num(opt_thresh, "ID", 2)
    t_up = tier.upper()
    if "HIGH" in t_up:
        return (
            "RISIKO TINGGI / PNEUMONIA TERDETEKSI",
            "Indikasi kuat konsolidasi pneumonia. Prioritaskan evaluasi dokter dan rujukan segera.",
            "Risiko Tinggi",
        )
    if "SUSPECT" in t_up or "PRIORITY" in t_up:
        return (
            "SUSPEK / TINJAUAN PRIORITAS",
            f"Probabilitas melampaui ambang triase F2 ({thresh_id}). Disarankan verifikasi klinis sekunder.",
            "Suspek",
        )
    return (
        "NORMAL / RISIKO RENDAH",
        "Tidak terdeteksi pola infiltrat pneumonia signifikan di atas ambang triase klinis.",
        "Normal",
    )


def compute_upload_signature(uploaded_file) -> str:
    """Builds a collision-resistant O(1) source ID from name, byte size, and head/tail slices."""
    raw_bytes = uploaded_file.getvalue()
    sample_slice = raw_bytes[:4096] + raw_bytes[-4096:]
    digest = hashlib.blake2b(sample_slice, digest_size=8).hexdigest()
    return f"upload:{uploaded_file.name}:{uploaded_file.size}:{digest}"


def decode_uploaded_bgr(uploaded_file) -> np.ndarray:
    """Safely decodes RGBA screenshots and 16-bit medical PNGs into 8-bit BGR arrays."""
    pil_raw = ImageOps.exif_transpose(Image.open(uploaded_file))

    if pil_raw.mode in ("RGBA", "LA") or (
        pil_raw.mode == "P" and "transparency" in pil_raw.info
    ):
        rgba = pil_raw.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        pil_img = Image.alpha_composite(bg, rgba).convert("RGB")
    elif pil_raw.mode.startswith("I"):
        arr16 = np.array(pil_raw, dtype=np.float32)
        min_v, max_v = float(arr16.min()), float(arr16.max())
        if max_v > min_v:
            arr8 = np.clip((arr16 - min_v) / (max_v - min_v) * 255.0, 0, 255).astype(
                np.uint8
            )
        else:
            arr8 = np.zeros_like(arr16, dtype=np.uint8)
        pil_img = Image.fromarray(arr8, mode="L").convert("RGB")
    else:
        pil_img = pil_raw.convert("RGB")

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ==========================================
# VISUALIZATION, UI & EXPORT HELPERS
# ==========================================
def render_responsive_image(img: np.ndarray, caption: str):
    """Renders image cleanly across both newer (width='stretch') and older Streamlit versions."""
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)


def render_full_width_button(label: str, key: str) -> bool:
    """Renders full-width button without deprecation warnings across Streamlit versions."""
    try:
        return st.button(label, key=key, width="stretch")
    except TypeError:
        return st.button(label, key=key, use_container_width=True)


def render_full_width_download(
    label: str, data, file_name: str, mime: str, key: str
):
    """Renders full-width download button without deprecation warnings across Streamlit versions."""
    try:
        st.download_button(
            label=label,
            data=data,
            file_name=file_name,
            mime=mime,
            key=key,
            width="stretch",
        )
    except TypeError:
        st.download_button(
            label=label,
            data=data,
            file_name=file_name,
            mime=mime,
            key=key,
            use_container_width=True,
        )


def render_triage_zone_bar(prob: float, opt_thresh: float, t: dict, lang: str):
    """Renders a lightweight HTML/CSS 3-zone probability bar with locale-consistent decimals."""
    t_low = float(np.clip(opt_thresh, 0.05, 0.45))
    t_high = 0.50
    w_normal = t_low * 100.0
    w_suspect = (t_high - t_low) * 100.0
    w_high = (1.0 - t_high) * 100.0
    marker_pct = float(np.clip(prob * 100.0, 1.0, 99.0))

    s_zero = fmt_num(0.0, lang, 2)
    s_low = fmt_num(t_low, lang, 2)
    s_high_prev = fmt_num(t_high - 0.01, lang, 2)
    s_high = fmt_num(t_high, lang, 2)
    s_one = fmt_num(1.0, lang, 2)

    html = f"""
    <div style="margin: 0.4rem 0 1.0rem 0; font-family: sans-serif;">
        <div style="display: flex; justify-content: space-between; font-size: 0.80rem; opacity: 0.85; margin-bottom: 4px;">
            <span><strong>{t['bar_title']}</strong></span>
            <span>{t['bar_scale'].format(zero=s_zero, t_low=s_low, t_high=s_high, one=s_one)}</span>
        </div>
        <div style="position: relative; width: 100%; height: 22px; border-radius: 6px; overflow: hidden; display: flex; border: 1px solid rgba(128,128,128,0.35);">
            <div style="width: {w_normal:.1f}%; background: #2e7d32; opacity: 0.85;" title="Normal"></div>
            <div style="width: {w_suspect:.1f}%; background: #ed6c02; opacity: 0.85;" title="Suspect"></div>
            <div style="width: {w_high:.1f}%; background: #d32f2f; opacity: 0.85;" title="High Risk"></div>
            <div style="position: absolute; left: {marker_pct:.2f}%; top: 0; bottom: 0; width: 3px; background: #ffffff; box-shadow: 0 0 4px #000000;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.76rem; margin-top: 4px; opacity: 0.9;">
            <span style="color: #4caf50;">■ {t['bar_zone_normal'].format(t_low=s_low)}</span>
            <span style="color: #ff9800;">■ {t['bar_zone_suspect'].format(t_low=s_low, t_high_prev=s_high_prev)}</span>
            <span style="color: #ef5350;">■ {t['bar_zone_high'].format(t_high=s_high)}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def build_composite_png(preprocessed_rgb: np.ndarray, gradcam_overlay: np.ndarray) -> bytes:
    """Combines CLAHE radiograph and Grad-CAM++ overlay side-by-side into PNG bytes."""
    divider = np.full((preprocessed_rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    combined = np.hstack([preprocessed_rgb, divider, gradcam_overlay])
    buf = io.BytesIO()
    Image.fromarray(combined).save(buf, format="PNG")
    return buf.getvalue()


def build_triage_report_txt(
    scan_ts: str,
    source_name: str,
    prob: float,
    opt_thresh: float,
    tier_disp: str,
    desc_disp: str,
    conf: int,
    latency_ms: float,
    cam_cutoff: float,
    strict_ood: bool,
    lang: str,
) -> str:
    """Builds a structured plain-text clinical triage summary note with frozen scan timestamp."""
    prob_str = fmt_pct(prob, lang, 2)
    thresh_str = fmt_num(opt_thresh, lang, 2)
    cutoff_str = fmt_num(cam_cutoff, lang, 2)

    if lang == "ID":
        return (
            "============================================================\n"
            "PNEUMOSCAN-X · LAPORAN TRIASE KLINIS RONTGEN DADA\n"
            "============================================================\n"
            f"Waktu Pemindaian     : {scan_ts}\n"
            f"Nama Berkas          : {source_name}\n"
            f"Arsitektur Model     : MobileNetV2 (PneumoScan-X)\n"
            f"Guard OOD RGB        : {'Aktif' if strict_ood else 'Nonaktif'}\n"
            f"Ambang Grad-CAM++    : {cutoff_str}\n"
            "------------------------------------------------------------\n"
            f"Probabilitas Pneumonia : {prob_str}\n"
            f"Ambang Triase (F2)     : {thresh_str}\n"
            f"Kategori Triase        : {tier_disp}\n"
            f"Keyakinan Zona         : {conf}%\n"
            f"Latensi Inferensi      : {latency_ms:.0f} ms\n"
            "------------------------------------------------------------\n"
            "REKOMENDASI TINDAK LANJUT:\n"
            f"{desc_disp}\n\n"
            "PENAFIAN PROTOTIPE:\n"
            "PneumoScan-X adalah prototipe pendukung keputusan klinis untuk\n"
            "prioritas triase dan BUKAN pengganti diagnosis dokter radiologi.\n"
            "============================================================\n"
        )
    return (
        "============================================================\n"
        "PNEUMOSCAN-X · CLINICAL CHEST X-RAY TRIAGE NOTE\n"
        "============================================================\n"
        f"Scan Timestamp       : {scan_ts}\n"
        f"Radiograph File      : {source_name}\n"
        f"Model Architecture   : MobileNetV2 (PneumoScan-X)\n"
        f"RGB OOD Guard        : {'Enabled' if strict_ood else 'Disabled'}\n"
        f"Grad-CAM++ Cutoff    : {cutoff_str}\n"
        "------------------------------------------------------------\n"
        f"Pneumonia Probability : {prob_str}\n"
        f"Triage Cutoff (F2)    : {thresh_str}\n"
        f"Assigned Triage Tier  : {tier_disp}\n"
        f"Zone Confidence       : {conf}%\n"
        f"Inference Latency     : {latency_ms:.0f} ms\n"
        "------------------------------------------------------------\n"
        "CLINICAL ACTION RECOMMENDATION:\n"
        f"{desc_disp}\n\n"
        "PROTOTYPE DISCLAIMER:\n"
        "PneumoScan-X is a clinical decision-support prototype for triage\n"
        "prioritization and is NOT a substitute for radiological diagnosis.\n"
        "============================================================\n"
    )


def create_synthetic_ood_bgr() -> np.ndarray:
    """Creates a synthetic non-grayscale RGB color image in memory to test the OOD Guard."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :75] = (220, 40, 40)      # Strong Blue block (BGR)
    img[:, 75:150] = (40, 200, 40)   # Strong Green block (BGR)
    img[:, 150:] = (40, 40, 230)     # Strong Red block (BGR)
    cv2.putText(
        img, "OOD COLOR TEST", (24, 118),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    return img


# ==========================================
# MODEL & EXPLAINABILITY HELPERS
# ==========================================
@st.cache_resource(show_spinner="Loading PneumoScan-X MobileNetV2 weights...")
def get_model():
    if not os.path.exists(WEIGHTS_PATH):
        tmp_path = f"{WEIGHTS_PATH}.tmp"
        try:
            with urllib.request.urlopen(GITHUB_WEIGHTS_URL, timeout=30) as resp, open(
                tmp_path, "wb"
            ) as out_file:
                shutil.copyfileobj(resp, out_file)
            os.replace(tmp_path, WEIGHTS_PATH)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    model, opt_thresh = load_trained_model(WEIGHTS_PATH, DEVICE)
    model.eval()
    return model, float(opt_thresh)


def run_inference_and_raw_gradcam(
    img_np: np.ndarray,
    model: torch.nn.Module,
    opt_thresh: float,
):
    """Runs forward pass + computes raw unthresholded Grad-CAM++ activation map."""
    t0 = time.perf_counter()
    scan_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tensor = TF_EVAL(img_np).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    tier, desc, conf = format_clinical_triage(prob, threshold=opt_thresh)

    cam_input = tensor.clone().detach().requires_grad_(True)
    target_block = model.features[-1]
    prev_grad_states = [p.requires_grad for p in target_block.parameters()]

    try:
        for param in target_block.parameters():
            param.requires_grad_(True)

        with torch.enable_grad():
            with GradCAMPlusPlus(model=model, target_layers=[target_block]) as cam:
                raw_g_cam = cam(
                    input_tensor=cam_input, targets=[ClassifierOutputTarget(1)]
                )[0, :].astype(np.float32)
    finally:
        model.zero_grad(set_to_none=True)
        for param, prev_state in zip(target_block.parameters(), prev_grad_states):
            param.requires_grad_(prev_state)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return prob, tier, desc, conf, raw_g_cam, latency_ms, scan_ts


def render_cam_overlay(
    preprocessed_rgb: np.ndarray, raw_g_cam: np.ndarray, cam_cutoff: float
) -> np.ndarray:
    """Applies threshold masking and JET colormap blending in <3 ms without rerunning PyTorch."""
    g_cam_masked = np.where(raw_g_cam < cam_cutoff, 0.0, raw_g_cam)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * g_cam_masked), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    base_rgb = preprocessed_rgb.astype(np.float32)
    alpha = (g_cam_masked > 0.0).astype(np.float32)[..., None] * 0.42
    return np.clip((1.0 - alpha) * base_rgb + alpha * heatmap_rgb, 0, 255).astype(np.uint8)


def discover_sample_presets() -> dict:
    """Discovers sample images inside ./sample_images using language-independent keys."""
    presets = {}
    sample_dir = "sample_images"
    if os.path.isdir(sample_dir):
        for fname in sorted(os.listdir(sample_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                presets[fname] = os.path.join(sample_dir, fname)
    presets[SYNTHETIC_OOD_TOKEN] = SYNTHETIC_OOD_TOKEN
    return presets


def format_preset_option(preset_key: str, t: dict) -> str:
    """Formats stable preset keys into localized display labels."""
    if preset_key == SYNTHETIC_OOD_TOKEN:
        return t["synthetic_ood_label"]
    label = preset_key.rsplit(".", 1)[0].replace("_", " ").title()
    return f"{label} ({preset_key})"


# ==========================================
# MODEL INITIALIZATION
# ==========================================
model = None
opt_thresh = 0.12
model_load_error = None

try:
    model, opt_thresh = get_model()
except Exception as err:
    model_load_error = err

# ==========================================
# SIDEBAR: LANGUAGE, GUIDE, CALIBRATION & SPECS
# ==========================================
with st.sidebar:
    st.title("🫁 PneumoScan-X")
    lang = st.selectbox(
        "🌐 Language / Bahasa",
        options=["EN", "ID"],
        format_func=lambda k: "English (EN)" if k == "EN" else "Bahasa Indonesia (ID)",
        key="ui_lang_select",
    )
    t = I18N[lang]

    s_thresh_low = fmt_num(opt_thresh, lang, 2)
    s_thresh_high_prev = fmt_num(0.49, lang, 2)
    s_thresh_high = fmt_num(0.50, lang, 2)

    st.caption(t["subtitle"])
    st.markdown(
        f"**{t['team_label']}:** Erlangga Azhar & Affan Alhafiz Hikren  \n"
        f"**{t['school_label']}:** SMA Negeri 1 Kota Cilegon"
    )

    guide_markdown = t["guide_body"].format(
        t_low=s_thresh_low,
        t_high_prev=s_thresh_high_prev,
        t_high=s_thresh_high,
    )
    if hasattr(st, "dialog"):
        @st.dialog(t["guide_title"])
        def show_tutorial_dialog():
            st.markdown(guide_markdown)

        if render_full_width_button(t["guide_btn"], key="open_guide_btn"):
            show_tutorial_dialog()
    else:
        with st.expander(t["guide_btn"], expanded=False):
            st.markdown(guide_markdown)

    st.divider()

    st.subheader(t["calib_header"])
    cam_cutoff = st.slider(
        t["cam_slider"],
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
        help=t["cam_help"],
        key="cam_cutoff_slider",
    )
    strict_ood = st.checkbox(
        t["ood_checkbox"],
        value=True,
        help=t["ood_help"],
        key="strict_ood_checkbox",
    )

    st.divider()
    st.subheader(t["perf_header"])
    st.markdown(t["perf_bullets"].format(t_low=s_thresh_low))
    st.divider()
    st.info(t["disclaimer"])


# ==========================================
# MAIN INTERFACE
# ==========================================
st.title(t["main_title"])
st.markdown(t["main_desc"])

if model_load_error is not None:
    st.error(t["load_err"].format(path=WEIGHTS_PATH, err=model_load_error))
    st.stop()

presets = discover_sample_presets()
input_mode = st.radio(
    t["source_radio"],
    options=["upload", "preset"],
    format_func=lambda k: t["opt_upload"] if k == "upload" else t["opt_preset"],
    key="input_mode_radio",
    horizontal=True,
)

source_id = None
source_name = None
uploaded = None
sample_target = None

if input_mode == "upload":
    uploaded = st.file_uploader(
        t["uploader_label"],
        type=["jpg", "jpeg", "png"],
        key="xray_file_uploader",
    )
    if uploaded is not None:
        source_name = uploaded.name
        source_id = compute_upload_signature(uploaded)
else:
    selected_preset_key = st.selectbox(
        t["preset_select"],
        options=list(presets.keys()),
        format_func=lambda k: format_preset_option(k, t),
        key="sample_preset_select",
    )
    sample_target = presets[selected_preset_key]
    source_id = f"preset:{selected_preset_key}"
    source_name = (
        "synthetic_ood_color_test.png"
        if sample_target == SYNTHETIC_OOD_TOKEN
        else os.path.basename(sample_target)
    )

if source_id is None:
    st.session_state.pop("infer_cache_key", None)
    st.session_state.pop("infer_cache_val", None)
else:
    st.divider()
    try:
        # Cache raw Grad-CAM++ map by (source_id, strict_ood) so slider & download reruns take <3 ms
        infer_cache_key = (source_id, bool(strict_ood))

        if st.session_state.get("infer_cache_key") != infer_cache_key:
            try:
                if input_mode == "upload":
                    raw_bgr = decode_uploaded_bgr(uploaded)
                elif sample_target == SYNTHETIC_OOD_TOKEN:
                    raw_bgr = create_synthetic_ood_bgr()
                else:
                    raw_bgr = cv2.imread(sample_target)
                    if raw_bgr is None:
                        raise RuntimeError(t["decode_err"].format(path=sample_target))

                preprocessed_rgb = preprocess_xray_cv(raw_bgr, validate_xray=strict_ood)
                infer_tuple = run_inference_and_raw_gradcam(
                    preprocessed_rgb, model, opt_thresh
                )
                st.session_state["infer_cache_val"] = ("ok", (preprocessed_rgb, *infer_tuple))
            except ValueError as ood_err:
                st.session_state["infer_cache_val"] = ("ood", str(ood_err))
            except Exception as exc:
                st.session_state["infer_cache_val"] = ("err", str(exc))

            st.session_state["infer_cache_key"] = infer_cache_key

        cache_status, cache_payload = st.session_state["infer_cache_val"]
        if cache_status == "ood":
            raise ValueError(cache_payload)
        if cache_status == "err":
            raise RuntimeError(cache_payload)

        (
            preprocessed_rgb,
            prob,
            tier_raw,
            desc_raw,
            conf,
            raw_g_cam,
            latency_ms,
            scan_ts,
        ) = cache_payload

        # Fast real-time threshold masking & heatmap overlay
        gradcam_overlay = render_cam_overlay(preprocessed_rgb, raw_g_cam, float(cam_cutoff))

        # Localize triage tier & description dynamically with opt_thresh
        tier_disp, desc_disp, zone_short = localize_triage_output(
            tier_raw, desc_raw, lang, opt_thresh
        )

        # 3-Tier Clinical Triage Banner
        tier_upper = tier_raw.upper()
        if "HIGH" in tier_upper:
            st.error(f"🚨 **{tier_disp}** — {desc_disp}")
        elif "SUSPECT" in tier_upper or "PRIORITY" in tier_upper:
            st.warning(f"⚠️ **{tier_disp}** — {desc_disp}")
        else:
            st.success(f"✅ **{tier_disp}** — {desc_disp}")

        # Visual 3-Zone Triage Position Bar
        render_triage_zone_bar(prob, opt_thresh, t, lang)

        # Quantitative Triage Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t["metric_prob"], fmt_pct(prob, lang, 2))
        m2.metric(t["metric_cutoff"], fmt_num(opt_thresh, lang, 2))
        m3.metric(t["metric_zone"], zone_short)
        m4.metric(t["metric_conf"], f"{conf}%", delta=f"{latency_ms:.0f} ms")

        # Side-by-Side Visual Explainability
        cutoff_str = fmt_num(float(cam_cutoff), lang, 2)
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader(t["col1_title"])
            render_responsive_image(
                preprocessed_rgb,
                caption=t["col1_cap"].format(name=source_name),
            )

        with col_right:
            st.subheader(t["col2_title"])
            render_responsive_image(
                gradcam_overlay,
                caption=t["col2_cap"].format(cutoff=cutoff_str),
            )
            st.caption(t["cam_legend"].format(cutoff=cutoff_str))

        if "NORMAL" in tier_upper:
            st.caption(t["normal_note"].format(t_low=s_thresh_low))

        # Export Clinical Triage Artifacts
        st.divider()
        st.subheader(t["export_header"])
        clean_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.splitext(source_name)[0])
        composite_bytes = build_composite_png(preprocessed_rgb, gradcam_overlay)
        report_txt = build_triage_report_txt(
            scan_ts=scan_ts,
            source_name=source_name,
            prob=prob,
            opt_thresh=opt_thresh,
            tier_disp=tier_disp,
            desc_disp=desc_disp,
            conf=conf,
            latency_ms=latency_ms,
            cam_cutoff=float(cam_cutoff),
            strict_ood=strict_ood,
            lang=lang,
        )

        dl1, dl2 = st.columns(2)
        with dl1:
            render_full_width_download(
                label=t["dl_img_btn"],
                data=composite_bytes,
                file_name=f"PneumoScanX_{clean_stem}_composite.png",
                mime="image/png",
                key="dl_composite_btn",
            )
        with dl2:
            render_full_width_download(
                label=t["dl_txt_btn"],
                data=report_txt.encode("utf-8"),
                file_name=f"PneumoScanX_{clean_stem}_triage_note.txt",
                mime="text/plain",
                key="dl_report_btn",
            )

    except ValueError as ood_err:
        st.error(
            f"🛑 **{t['ood_err_title']}:** {ood_err}\n\n"
            f"{t['ood_err_body']}"
        )
    except Exception as runtime_err:
        st.error(t["runtime_err"].format(name=source_name, err=runtime_err))