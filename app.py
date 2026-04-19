"""Streamlit UI for training and inference of liver CNN model."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

from train_liver_cnn import (
    get_class_names,
    make_gradcam_heatmap,
    predict_single_image_detailed,
    preprocess_single_image,
    save_gradcam_overlay,
)


st.set_page_config(page_title="Liver Cancer AI", page_icon="🧪", layout="wide")


DATA_DIR_DEFAULT = Path("Liver_Dataset")
OUTPUT_DIR_DEFAULT = Path("outputs")
MODEL_PATH_DEFAULT = Path("liver_cancer_model.h5")


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&family=Manrope:wght@500;700;800&display=swap');

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(255, 255, 255, 0.65), rgba(255, 255, 255, 0) 30%),
                radial-gradient(circle at 90% 20%, rgba(255, 217, 179, 0.55), rgba(255, 255, 255, 0) 32%),
                linear-gradient(135deg, #f9fbff 0%, #eef5f4 50%, #fefaf2 100%);
            font-family: 'Cairo', sans-serif;
            color: #16363c;
        }

        p, label, span, div {
            color: #16363c;
        }

        h1, h2, h3, h4 {
            color: #0b2f35;
        }

        .main-title {
            font-family: 'Manrope', sans-serif;
            font-weight: 800;
            font-size: 2.1rem;
            color: #0b2f35;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #345b63;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }

        .soft-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(45, 74, 83, 0.12);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 10px 30px rgba(26, 66, 77, 0.08);
        }

        .metric-badge {
            background: linear-gradient(120deg, #173f4a, #2c6b68);
            color: #ffffff;
            border-radius: 14px;
            padding: 10px 14px;
            font-weight: 700;
            text-align: center;
        }

        .upload-hint {
            border: 1px solid rgba(22, 54, 60, 0.2);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.78);
            padding: 10px 12px;
            margin: 6px 0 12px 0;
            color: #16363c;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_class_names(class_names_json: Path) -> Optional[list[str]]:
    if not class_names_json.exists():
        return None
    return json.loads(class_names_json.read_text(encoding="utf-8"))


def resolve_inference_resources(
    preferred_model_path: Path,
    output_dir: Path,
    data_dir: Path,
) -> Tuple[Optional[Path], Optional[list[str]], str]:
    model_candidates = [
        preferred_model_path,
        output_dir / "best_model.keras",
        output_dir / "final_model.keras",
    ]

    model_path = next((path for path in model_candidates if path.exists()), None)

    class_json = output_dir / "class_names.json"
    class_names = read_class_names(class_json)
    if class_names is None and data_dir.exists():
        class_names = get_class_names(data_dir)

    if model_path is None:
        return None, class_names, "ملف النموذج غير موجود بعد التدريب."

    if class_names is None:
        return model_path, None, "تعذر إيجاد أسماء الفئات."

    return model_path, class_names, f"سيتم استخدام النموذج: {model_path}"


def find_healthy_class(probabilities: Dict[str, float]) -> Optional[str]:
    for cls_name in probabilities:
        lowered = cls_name.lower()
        if "healthy" in lowered or "normal" in lowered:
            return cls_name
    return None


def compute_disease_probability(probabilities: Dict[str, float]) -> Tuple[Optional[float], str]:
    healthy_class = find_healthy_class(probabilities)
    if healthy_class is not None:
        return 1.0 - probabilities[healthy_class], f"اعتمادًا على الفئة الصحية: {healthy_class}"

    if len(probabilities) == 2:
        classes_sorted = sorted(probabilities.keys())
        return probabilities[classes_sorted[1]], "تصنيف ثنائي بدون فئة Healthy صريحة"

    return None, "لا يمكن اشتقاق نسبة الإصابة تلقائيًا بدون فئة Healthy"


def run_training(
    data_dir: Path,
    output_dir: Path,
    model_path: Path,
    epochs: int,
    batch_size: int,
    patience: int,
) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        "-u",
        "train_liver_cnn.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--model-path",
        str(model_path),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--patience",
        str(patience),
    ]

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    logs: list[str] = []
    status_box = st.empty()
    progress_label = st.empty()
    progress_bar = st.progress(0)
    log_box = st.empty()
    status_box.info("بدأ التدريب... يتم تحديث السجل مباشرة")
    progress_label.caption(f"Epoch: 0/{epochs}")

    epoch_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)")

    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        clean_line = line.rstrip()
        if clean_line:
            logs.append(clean_line)
            epoch_match = epoch_pattern.search(clean_line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                percent = int((current_epoch / max(total_epochs, 1)) * 100)
                progress_bar.progress(min(percent, 100))
                progress_label.caption(f"Epoch: {current_epoch}/{total_epochs}")
            # Keep the latest chunk so UI stays responsive for long runs.
            log_box.code("\n".join(logs[-120:]), language="text")

    proc.wait()
    full_logs = "\n".join(logs)

    if proc.returncode == 0:
        progress_bar.progress(100)
        status_box.success("اكتمل التدريب بنجاح")
        return True, full_logs

    status_box.error("فشل التدريب. راجع السجل أدناه")
    return False, full_logs


def show_training_artifacts(output_dir: Path) -> None:
    curve_path = output_dir / "training_curves.png"
    cm_path = output_dir / "confusion_matrix.png"
    report_path = output_dir / "classification_report.txt"

    col1, col2 = st.columns(2)

    with col1:
        if curve_path.exists():
            st.image(str(curve_path), caption="Training Curves", use_container_width=True)

    with col2:
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)

    if report_path.exists():
        st.markdown("### Classification Report")
        st.code(report_path.read_text(encoding="utf-8"), language="text")


def show_prediction_section(model_path: Path, output_dir: Path, data_dir: Path) -> None:
    st.markdown("### الخطوة 3: تشخيص صورة مريض جديد")
    st.markdown(
        "<div class='upload-hint'>ارفع الصورة هنا ثم اضغط زر <b>تحليل الصورة</b> لعرض نسبة الإصابة والنتيجة.</div>",
        unsafe_allow_html=True,
    )

    resolved_model_path, class_names, resolution_note = resolve_inference_resources(model_path, output_dir, data_dir)
    if resolved_model_path is None or class_names is None:
        st.warning(f"غير جاهز للتنبؤ بعد: {resolution_note}")
        return

    st.caption(resolution_note)

    uploaded_file = st.file_uploader(
        "ارفع صورة طبية (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="patient_image",
    )

    use_gradcam = st.checkbox("إظهار Grad-CAM", value=True)

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", use_container_width=False, width=320)

    if st.button("تحليل الصورة", type="primary"):
        model = keras.models.load_model(resolved_model_path)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)

        pred = predict_single_image_detailed(model, tmp_path, class_names)
        probabilities = pred["probabilities"]

        disease_prob, note = compute_disease_probability(probabilities)

        left, right = st.columns(2)
        with left:
            st.markdown(
                f"<div class='metric-badge'>التشخيص المتوقع: {pred['predicted_class']}</div>",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f"<div class='metric-badge'>ثقة النموذج: {pred['confidence'] * 100:.2f}%</div>",
                unsafe_allow_html=True,
            )

        if disease_prob is not None:
            st.progress(float(disease_prob))
            st.info(f"نسبة الإصابة التقريبية: {disease_prob * 100:.2f}% | {note}")
        else:
            st.info(note)

        st.markdown("#### احتمالات كل فئة")
        prob_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in prob_items]
        values = [item[1] for item in prob_items]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(labels, values, color="#2c6b68")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if use_gradcam:
            image_batch = preprocess_single_image(tmp_path, (224, 224))
            heatmap = make_gradcam_heatmap(model, image_batch, last_conv_layer_name="last_conv_layer")
            gradcam_out = output_dir / "ui_gradcam_overlay.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_gradcam_overlay(tmp_path, heatmap, gradcam_out)
            st.image(str(gradcam_out), caption="Grad-CAM", use_container_width=False, width=350)


def main() -> None:
    apply_custom_style()

    st.markdown("<div class='main-title'>Liver Cancer AI Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>واجهة تدريب وتشخيص محلي لصور الكبد باستخدام CNN و TensorFlow</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("### الخطوة 1: إعدادات التدريب")
        st.caption("ملاحظة: التدريب يستمر حتى نهاية Epochs المحددة أو يتوقف مبكرًا عبر Early Stopping.")

        c1, c2, c3 = st.columns(3)
        with c1:
            data_dir = Path(st.text_input("Dataset Folder", str(DATA_DIR_DEFAULT)))
            epochs = st.slider("Epochs", 3, 80, 15)
        with c2:
            output_dir = Path(st.text_input("Output Folder", str(OUTPUT_DIR_DEFAULT)))
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=2)
        with c3:
            model_path = Path(st.text_input("Model Path", str(MODEL_PATH_DEFAULT)))
            patience = st.slider("Early Stopping Patience", 3, 20, 8)

        st.session_state["ui_data_dir"] = str(data_dir)
        st.session_state["ui_output_dir"] = str(output_dir)
        st.session_state["ui_model_path"] = str(model_path)

        if st.button("بدء التدريب", type="primary"):
            if not data_dir.exists():
                st.error(f"لم يتم العثور على مجلد البيانات: {data_dir}")
            else:
                ok, logs = run_training(data_dir, output_dir, model_path, epochs, batch_size, patience)

                if not ok:
                    st.error("حدث خطأ أثناء التدريب")

                if logs:
                    with st.expander("سجل التدريب"):
                        st.code(logs, language="text")

    with st.container(border=True):
        st.markdown("### الخطوة 2: نتائج التدريب")
        selected_output_dir = Path(st.session_state.get("ui_output_dir", str(OUTPUT_DIR_DEFAULT)))
        show_training_artifacts(selected_output_dir)

    with st.container(border=True):
        selected_model_path = Path(st.session_state.get("ui_model_path", str(MODEL_PATH_DEFAULT)))
        selected_output_dir = Path(st.session_state.get("ui_output_dir", str(OUTPUT_DIR_DEFAULT)))
        selected_data_dir = Path(st.session_state.get("ui_data_dir", str(DATA_DIR_DEFAULT)))
        show_prediction_section(selected_model_path, selected_output_dir, selected_data_dir)


if __name__ == "__main__":
    main()
