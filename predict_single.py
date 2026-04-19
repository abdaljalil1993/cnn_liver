"""Load trained model and run prediction on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorflow import keras

from train_liver_cnn import predict_single_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Single image inference for liver CNN")
    parser.add_argument("--model-path", type=str, default="liver_cancer_model.h5", help="Trained model path")
    parser.add_argument(
        "--class-names-json",
        type=str,
        default="outputs/class_names.json",
        help="Path to class_names.json generated during training",
    )
    parser.add_argument("--image-path", type=str, required=True, help="Path to image")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    class_json = Path(args.class_names_json)
    image_path = Path(args.image_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not class_json.exists():
        raise FileNotFoundError(f"Class names JSON not found: {class_json}")
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    class_names = json.loads(class_json.read_text(encoding="utf-8"))
    model = keras.models.load_model(model_path)

    pred = predict_single_image(model, image_path, class_names, (args.img_size, args.img_size))
    print(pred)


if __name__ == "__main__":
    main()
