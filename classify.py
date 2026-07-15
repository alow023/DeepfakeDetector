import torch
from torchvision import transforms
from models import load_efficientnet_with_cbam
from PIL import Image
import argparse

# Load your trained model
def load_model(model_path="models/best_model.pt", use_cbam=True):
    model = load_efficientnet_with_cbam(checkpoint_path=model_path, use_cbam=use_cbam, device="cpu")
    return model

# Preprocess and classify image
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    label = "FAKE" if pred == 1 else "REAL"
    print(f"\n🧠 Prediction: {label}")
    print(f"Real: {probs[0]:.3f} | Fake: {probs[1]:.3f}")

# Run from terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file (.jpg/.png)")
    args = parser.parse_args()

    model = load_model()
    predict_image(args.image_path, model)
