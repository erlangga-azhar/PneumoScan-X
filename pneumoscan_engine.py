import os
import re
import glob
import hashlib
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import recall_score, f1_score, fbeta_score, roc_auc_score, confusion_matrix
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# 0. REPRODUCIBILITY ENGINE
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. DEDUPLIKASI KETAT & PATIENT-AWARE SPLIT
# ==========================================
def extract_patient_id(filename):
    match_pneumonia = re.search(r'(person\d+|patient\d+)', filename, re.IGNORECASE)
    if match_pneumonia:
        return match_pneumonia.group(1).lower()

    match_normal = re.search(r'(IM-\d+)', filename, re.IGNORECASE)
    if match_normal:
        return match_normal.group(1).lower()

    return os.path.splitext(filename)[0].lower()

def prepare_dataset_splits(data_dir, test_size=0.15, val_size=0.15, random_state=42):
    filepaths, labels, patient_ids = [], [], []
    seen_filenames = set()
    seen_hashes = set()

    raw_paths = sorted(
        glob.glob(os.path.join(data_dir, "**", "*.[jJ][pP][gG]"), recursive=True) +
        glob.glob(os.path.join(data_dir, "**", "*.[jJ][pP][eE][gG]"), recursive=True) +
        glob.glob(os.path.join(data_dir, "**", "*.[pP][nN][gG]"), recursive=True)
    )

    for path in raw_paths:
        norm_path = path.replace("\\", "/")
        if "__MACOSX" in norm_path or "chest_xray/chest_xray" in norm_path:
            continue

        fname = os.path.basename(path)
        if fname.startswith(".") or fname in seen_filenames:
            continue

        with open(path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash in seen_hashes:
            continue

        parent_folder = os.path.basename(os.path.dirname(path)).upper()
        if parent_folder == "PNEUMONIA":
            label = 1
        elif parent_folder == "NORMAL":
            label = 0
        else:
            continue

        seen_filenames.add(fname)
        seen_hashes.add(file_hash)
        filepaths.append(path)
        labels.append(label)
        patient_ids.append(extract_patient_id(fname))

    assert len(filepaths) <= 5856, f"BUG: Masih terdeteksi duplikat ({len(filepaths)} file)."

    filepaths = np.array(filepaths)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(test_size + val_size), random_state=random_state)
    train_idx, temp_idx = next(gss1.split(filepaths, labels, groups=patient_ids))

    relative_val_size = val_size / (test_size + val_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - relative_val_size), random_state=random_state)
    val_idx, test_idx = next(gss2.split(filepaths[temp_idx], labels[temp_idx], groups=patient_ids[temp_idx]))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    print(f"Total Citra Unik Terverifikasi: {len(filepaths)}")
    print(f"Data Bersih -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Pasien Unik -> Train: {len(set(patient_ids[train_idx]))}, Val: {len(set(patient_ids[val_idx]))}, Test: {len(set(patient_ids[test_idx]))}")

    return (filepaths[train_idx], labels[train_idx]), \
           (filepaths[val_idx], labels[val_idx]), \
           (filepaths[test_idx], labels[test_idx])

# ==========================================
# 2. PREPROCESSING MULTI-INPUT & OOD GUARD
# ==========================================
def _assert_grayscale_xray(img_bgr, color_diff_thresh=15.0):
    """Menolak foto berwarna non-medis sebelum masuk ke model."""
    if img_bgr is not None and len(img_bgr.shape) == 3 and img_bgr.shape[2] >= 3:
        b, g, r = cv2.split(img_bgr[:, :, :3])
        diff_rg = np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
        diff_gb = np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
        if max(diff_rg, diff_gb) > color_diff_thresh:
            raise ValueError("Input ditolak: Citra terdeteksi sebagai foto berwarna, bukan X-Ray dada grayscale.")

def preprocess_xray_cv(img_input, crop_pct=0.05, clip_limit=1.5, validate_xray=True):
    """
    Safe Center Crop (5%) + CLAHE Enhancement (clipLimit 1.5).
    Mendukung str, bytes, BytesIO (Streamlit UploadedFile), atau np.ndarray.
    """
    if isinstance(img_input, (str, os.PathLike)):
        raw = cv2.imread(str(img_input), cv2.IMREAD_COLOR)
    elif isinstance(img_input, bytes):
        raw = cv2.imdecode(np.frombuffer(img_input, np.uint8), cv2.IMREAD_COLOR)
    elif hasattr(img_input, 'read'):
        bytes_data = img_input.read()
        if hasattr(img_input, 'seek'):
            img_input.seek(0)
        raw = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(img_input, np.ndarray):
        raw = img_input
    else:
        raise ValueError(f"Format input tidak didukung: {type(img_input)}")

    if raw is None:
        raise ValueError("Gagal melakukan decoding citra X-Ray.")

    if validate_xray:
        _assert_grayscale_xray(raw)

    img = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if len(raw.shape) == 3 else raw

    h, w = img.shape
    ch, cw = int(h * crop_pct), int(w * crop_pct)
    cropped = img[ch:h-ch, cw:w-cw]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(cropped)

    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return cv2.resize(rgb, (224, 224))

class XRayDataset(Dataset):
    def __init__(self, filepaths, labels, is_train=False):
        self.filepaths = filepaths
        self.labels = labels
        self.is_train = is_train

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=10) if is_train else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_np = preprocess_xray_cv(self.filepaths[idx], validate_xray=False)
        tensor_img = self.transform(img_np)
        return tensor_img, torch.tensor(self.labels[idx], dtype=torch.long), self.filepaths[idx]

# ==========================================
# 3. MODEL BUILDER & HANDOVER HELPER
# ==========================================
def build_mobilenet_v2(num_classes=2, unfreeze_top_blocks=True):
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_top_blocks:
        for param in model.features[14:].parameters():
            param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return model

def load_trained_model(pth_path, device):
    model = build_mobilenet_v2(unfreeze_top_blocks=False).to(device)
    checkpoint = torch.load(pth_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        threshold = checkpoint.get('optimal_threshold', 0.12)
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.12

    model.eval()
    return model, threshold

# ==========================================
# 4. OPTIMAL THRESHOLD TUNING & CLINICAL TRIAGE
# ==========================================
def find_optimal_threshold(val_targets, val_probs):
    best_thresh = 0.5
    best_f2 = 0.0
    for thresh in np.arange(0.05, 0.60, 0.01):
        preds = (val_probs >= thresh).astype(int)
        f2 = fbeta_score(val_targets, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = thresh
    return float(round(best_thresh, 2)), float(best_f2)

def format_clinical_triage(prob, threshold=0.12):
    """
    Memetakan probabilitas mentah ke 3 zona triase klinis untuk UI Streamlit.
    """
    if prob < threshold:
        confidence = (1.0 - (prob / max(threshold, 1e-6))) * 100.0
        return "NORMAL / LOW RISK", "Tidak terdeteksi pola opasitas pneumonia signifikan.", round(confidence, 1)
    elif prob < 0.50:
        confidence = 50.0 + ((prob - threshold) / (0.50 - threshold)) * 30.0
        return "SUSPECT / PRIORITY REVIEW", "Terdeteksi indikasi dini/batas. Prioritaskan evaluasi dokter.", round(confidence, 1)
    else:
        confidence = 80.0 + ((prob - 0.50) / 0.50) * 20.0
        return "HIGH PROBABILITY PNEUMONIA", "Terdeteksi pola konsolidasi kuat. Segera rujuk penanganan klinis.", round(confidence, 1)

# ==========================================
# 5. TRAINING & EVALUATION ENGINE
# ==========================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= threshold).astype(int)

    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    f2 = fbeta_score(all_targets, all_preds, beta=2, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)

    return rec, f1, f2, auc, cm, all_probs, all_targets

# ==========================================
# 6. GRAD-CAM++ EXPORTER (ROBUST FOR UI DEMO)
# ==========================================
def generate_gradcam_overlay(model, img_input, device, threshold=0.12, cam_cutoff=0.20):
    model.eval()
    img_np = preprocess_xray_cv(img_input, validate_xray=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_np).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()

    is_pneumonia = prob >= threshold
    rgb_float = img_np.astype(np.float32) / 255.0

    if not is_pneumonia:
        cam_image = (rgb_float * 255).astype(np.uint8)
    else:
        target_layers = [model.features[-1]]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])[0, :]

        grayscale_cam[grayscale_cam < cam_cutoff] = 0.0
        cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return cam_image, prob, is_pneumonia

# ==========================================
# 7. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    set_seed(42)
    DATA_DIR = "./dataset_raw"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan Device: {DEVICE}")

    (train_files, train_lbls), (val_files, val_lbls), (test_files, test_lbls) = prepare_dataset_splits(DATA_DIR)

    train_ds = XRayDataset(train_files, train_lbls, is_train=True)
    val_ds = XRayDataset(val_files, val_lbls, is_train=False)
    test_ds = XRayDataset(test_files, test_lbls, is_train=False)

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=use_pin)

    n_normal = np.sum(train_lbls == 0)
    n_pneumonia = np.sum(train_lbls == 1)
    weights = torch.tensor([n_pneumonia / n_normal, 1.0], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_mobilenet_v2(unfreeze_top_blocks=True).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.features[14:].parameters() if p.requires_grad], 'lr': 1e-5},
        {'params': [p for p in model.classifier.parameters() if p.requires_grad], 'lr': 1e-4}
    ], weight_decay=1e-2)

    BEST_F2 = 0.0
    BEST_THRESH = 0.5
    PATIENCE = 3
    patience_counter = 0

    for epoch in range(1, 16):
        loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        _, _, _, auc, _, val_probs, val_targets = evaluate(model, val_loader, DEVICE, threshold=0.5)

        opt_thresh, val_f2 = find_optimal_threshold(val_targets, val_probs)
        rec, f1, _, _, cm, _, _ = evaluate(model, val_loader, DEVICE, threshold=opt_thresh)

        print(f"Epoch {epoch:2d} | Loss: {loss:.4f} | Opt Thresh: {opt_thresh:.2f} | Val Recall: {rec:.4f} | Val F2: {val_f2:.4f} | Val AUC: {auc:.4f}")

        if val_f2 > BEST_F2:
            BEST_F2 = val_f2
            BEST_THRESH = opt_thresh
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimal_threshold': BEST_THRESH
            }, "best_mobilenet_pneumonia.pth")
            print(f"--> Checkpoint disimpan (Opt Threshold: {BEST_THRESH:.2f}, Val F2: {BEST_F2:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"--> Early stopping dipicu pada Epoch {epoch}. Training dihentikan.")
                break

    model, optimal_thresh = load_trained_model("best_mobilenet_pneumonia.pth", DEVICE)
    rec, f1, f2, auc, cm, _, _ = evaluate(model, test_loader, DEVICE, threshold=optimal_thresh)
    print(f"\n=== HASIL EVALUASI TEST SET (Threshold: {optimal_thresh:.2f}) ===")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"F2-Score: {f2:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
