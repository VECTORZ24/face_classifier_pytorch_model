"""
test.py
Run inference with the trained model on a single image or a folder of images.

Usage:
    python test.py --model face_classifier.pt --image path/to/photo.jpg
    python test.py --model face_classifier.pt --folder path/to/folder --threshold 0.6
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def get_transform(image_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def predict_image(model, classes, transform, image_path, device, threshold=0.0):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, dim=0)

    predicted_class = classes[pred_idx.item()]
    confidence = confidence.item()

    # Optional extra safety net: if confidence is low even for a "known" class,
    # fall back to unidentified. Useful if your unidentified training data
    # was limited and the model is overconfident.
    if threshold > 0 and confidence < threshold and predicted_class != "unidentified":
        predicted_class = "unidentified (low confidence)"

    return predicted_class, confidence, {c: round(p.item(), 4) for c, p in zip(classes, probs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="face_classifier.pt")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images to classify")
    parser.add_argument("--threshold", type=float, default=0.0,
                         help="Minimum confidence to trust a Ghaith/Aziz prediction "
                              "(e.g. 0.6). Below this, falls back to unidentified.")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide either --image or --folder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.model, device)
    transform = get_transform()

    if args.image:
        pred, conf, probs = predict_image(model, classes, transform, args.image, device, args.threshold)
        print(f"{args.image} -> {pred} (confidence: {conf:.3f})")
        print(f"  full probabilities: {probs}")

    if args.folder:
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for fname in sorted(os.listdir(args.folder)):
            if fname.lower().endswith(valid_ext):
                fpath = os.path.join(args.folder, fname)
                pred, conf, probs = predict_image(model, classes, transform, fpath, device, args.threshold)
                print(f"{fname:40s} -> {pred:25s} (confidence: {conf:.3f})")


if __name__ == "__main__":
    main()