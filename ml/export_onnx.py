"""
ml/export_onnx.py
Export heart and lung MobileNetV2 checkpoints to ONNX format for RPi deployment.

Usage:
    python ml/export_onnx.py
Outputs:
    ml/data/cnn_model_heart.onnx
    ml/data/cnn_model_lung.onnx
"""

import os
import json
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MODELS = {
    "heart": os.path.join(DATA_DIR, "cnn_model_heart.pth"),
    "lung":  os.path.join(DATA_DIR, "cnn_model_lung.pth"),
}


def build_mobilenet(n_classes: int) -> nn.Module:
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, n_classes)
    return model


def export(name: str, pth_path: str):
    print(f"Loading {name} model from {pth_path} ...")
    ckpt = torch.load(pth_path, map_location="cpu")

    n_classes = ckpt["n_classes"]
    img_size  = ckpt["img_size"]

    model = build_mobilenet(n_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, img_size, img_size)
    onnx_path = pth_path.replace(".pth", ".onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    meta = {
        "n_classes":   n_classes,
        "label_names": ckpt["label_names"],
        "target_sr":   ckpt["target_sr"],
        "n_mels":      ckpt["n_mels"],
        "hop_length":  ckpt["hop_length"],
        "img_size":    img_size,
    }
    json_path = onnx_path.replace(".onnx", ".json")
    with open(json_path, "w") as f:
        json.dump(meta, f)

    print(f"  -> {onnx_path}  ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    print(f"     classes={ckpt['label_names']}  img_size={img_size}")
    print(f"  -> {json_path}")


if __name__ == "__main__":
    for name, path in MODELS.items():
        if os.path.exists(path):
            export(name, path)
        else:
            print(f"SKIP {name}: {path} not found")
