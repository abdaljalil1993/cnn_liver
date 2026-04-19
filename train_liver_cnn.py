"""Train and evaluate a CNN for liver cancer image classification.

This script expects a local dataset folder with one subdirectory per class:
Liver_Dataset/
    ClassA/
    ClassB/
    ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers


def get_class_names(data_dir: Path) -> List[str]:
    """Get sorted class names from dataset subfolders."""
    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class folders found in: {data_dir}")
    return class_names


def get_label_mode(num_classes: int) -> str:
    """Select Keras label mode based on class count."""
    return "binary" if num_classes == 2 else "categorical"


def create_datasets(
    data_dir: Path,
    img_size: Tuple[int, int],
    batch_size: int,
    seed: int,
    val_split: float,
    label_mode: str,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Create train/validation datasets from directory."""
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=True,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=False,
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """Build a CNN model for binary or multi-class classification."""
    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomFlip("horizontal"),
        ],
        name="augmentation",
    )

    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="last_conv_layer")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "categorical_crossentropy"

    model = keras.Model(inputs=inputs, outputs=outputs, name="liver_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history: keras.callbacks.History, output_dir: Path) -> None:
    """Plot and save training/validation accuracy and loss curves."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="train_accuracy")
    axes[0].plot(history.history["val_accuracy"], label="val_accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def evaluate_model(
    model: keras.Model,
    val_ds: tf.data.Dataset,
    class_names: List[str],
    output_dir: Path,
) -> Dict[str, str]:
    """Evaluate model and save classification report and confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred_prob = model.predict(val_ds, verbose=1)
    y_true_batches = []

    for _, labels in val_ds:
        y_true_batches.append(labels.numpy())

    if len(class_names) == 2:
        y_true = np.concatenate(y_true_batches).astype(int).reshape(-1)
        y_pred = (y_pred_prob.reshape(-1) >= 0.5).astype(int)
    else:
        y_true = np.concatenate(y_true_batches)
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred_prob, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    return {
        "classification_report": str(report_path),
        "confusion_matrix": str(cm_path),
    }


def preprocess_single_image(image_path: Path, img_size: Tuple[int, int]) -> np.ndarray:
    """Load and preprocess one image for model inference."""
    img = keras.utils.load_img(image_path, target_size=img_size)
    arr = keras.utils.img_to_array(img)
    arr = arr.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_single_image(
    model: keras.Model,
    image_path: Path,
    class_names: List[str],
    img_size: Tuple[int, int] = (224, 224),
) -> Dict[str, float | str]:
    """Predict class and confidence for a single image."""
    detailed = predict_single_image_detailed(model, image_path, class_names, img_size)
    return {
        "predicted_class": str(detailed["predicted_class"]),
        "confidence": float(detailed["confidence"]),
    }


def predict_single_image_detailed(
    model: keras.Model,
    image_path: Path,
    class_names: List[str],
    img_size: Tuple[int, int] = (224, 224),
) -> Dict[str, object]:
    """Predict class, confidence, and per-class probabilities for one image."""
    image_batch = preprocess_single_image(image_path, img_size)
    probs = model.predict(image_batch, verbose=0)[0]

    if len(class_names) == 2:
        score = float(np.squeeze(probs))
        class_probs = np.array([1.0 - score, score], dtype=np.float32)
    else:
        class_probs = np.array(probs, dtype=np.float32)

    pred_idx = int(np.argmax(class_probs))
    confidence = float(class_probs[pred_idx])
    prob_map = {name: float(class_probs[i]) for i, name in enumerate(class_names)}

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": confidence,
        "probabilities": prob_map,
    }


def make_gradcam_heatmap(
    model: keras.Model,
    image_batch: np.ndarray,
    last_conv_layer_name: str,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for one preprocessed image batch."""
    grad_model = keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_overlay(
    image_path: Path,
    heatmap: np.ndarray,
    output_path: Path,
    alpha: float = 0.4,
) -> None:
    """Overlay Grad-CAM heatmap on original image and save."""
    original = Image.open(image_path).convert("RGB")
    original_arr = np.array(original).astype("float32")

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap_uint8).resize(original.size, resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img) / 255.0

    cmap = plt.get_cmap("jet")
    jet = cmap(heatmap_resized)[..., :3]
    jet = (jet * 255.0).astype("float32")

    overlay = jet * alpha + original_arr * (1.0 - alpha)
    overlay = np.clip(overlay, 0, 255).astype("uint8")

    Image.fromarray(overlay).save(output_path)


def train(args: argparse.Namespace) -> None:
    """Main training and evaluation workflow."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    class_names = get_class_names(data_dir)
    num_classes = len(class_names)
    label_mode = get_label_mode(num_classes)

    print(f"Detected {num_classes} classes: {class_names}")

    train_ds, val_ds, class_names = create_datasets(
        data_dir=data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        seed=args.seed,
        val_split=args.val_split,
        label_mode=label_mode,
    )

    model = build_model((args.img_size, args.img_size, 3), num_classes)
    model.summary()

    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_curves(history, output_dir)

    results = evaluate_model(model, val_ds, class_names, output_dir)
    print("Saved evaluation artifacts:")
    print(results)

    class_json = output_dir / "class_names.json"
    class_json.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Saved class labels to: {class_json}")

    try:
        model.save(model_path)
        print(f"Saved final model to: {model_path}")
    except Exception as exc:
        fallback_model_path = output_dir / "final_model.keras"
        model.save(fallback_model_path)
        print(f"Failed to save model to {model_path}: {exc}")
        print(f"Saved fallback model to: {fallback_model_path}")

    if args.sample_image:
        sample_path = Path(args.sample_image)
        prediction = predict_single_image(model, sample_path, class_names, (args.img_size, args.img_size))
        print(f"Sample prediction for {sample_path}: {prediction}")

    if args.gradcam_image:
        gradcam_path = Path(args.gradcam_image)
        image_batch = preprocess_single_image(gradcam_path, (args.img_size, args.img_size))
        heatmap = make_gradcam_heatmap(model, image_batch, last_conv_layer_name="last_conv_layer")
        gradcam_output = output_dir / "gradcam_overlay.png"
        save_gradcam_overlay(gradcam_path, heatmap, gradcam_output)
        print(f"Saved Grad-CAM overlay to: {gradcam_output}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CNN for liver image classification")
    parser.add_argument("--data-dir", type=str, default="Liver_Dataset", help="Path to dataset root")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to save metrics and plots")
    parser.add_argument(
        "--model-path",
        type=str,
        default="liver_cancer_model.h5",
        help="Output path for trained model",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument(
        "--sample-image",
        type=str,
        default="",
        help="Optional image path for a post-training prediction",
    )
    parser.add_argument(
        "--gradcam-image",
        type=str,
        default="",
        help="Optional image path for Grad-CAM overlay",
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
