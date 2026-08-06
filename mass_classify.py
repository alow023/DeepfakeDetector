import os
import argparse
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# Load trained model
def load_model(model_path="models/best_model-v3.pt"):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# Predict one image
def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    label = "FAKE" if pred == 1 else "REAL"

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Prediction: {label}")
    print(f"Real: {probs[0]:.3f}")
    print(f"Fake: {probs[1]:.3f}")


# Predict all images in a folder
def predict_folder(folder_path, model):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_extensions)
    ])

    if len(image_files) == 0:
        print("No images found in the folder.")
        return

    print(f"\nFound {len(image_files)} images.\n")

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        predict_image(image_path, model)


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a single image or all images in a folder."
    )

    parser.add_argument(
        "input_path",
        help="Path to an image file or a folder containing images"
    )

    args = parser.parse_args()

    model = load_model()

    if os.path.isdir(args.input_path):
        predict_folder(args.input_path, model)
    elif os.path.isfile(args.input_path):
        predict_image(args.input_path, model)
    else:
        print("Error: The specified path does not exist.")