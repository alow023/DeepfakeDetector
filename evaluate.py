import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ======================================================
# CONFIGURATION
# ======================================================

MODEL_PATH = "models/best_model-v3.pt"

TEST_FOLDER = r"C:\arissa\Deepfake detection\Data\test"

OUTPUT_FILE = "prediction_results.xlsx"

# ======================================================
# LOAD YOUR TRAINED MODEL (.ckpt)
# ======================================================

MODEL_PATH = "models/best_model.ckpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the same model architecture used during training
model = efficientnet_b0()

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(
        model.classifier[1].in_features,
        2
    )
)

# Load Lightning checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Remove the "model." prefix from all keys
state_dict = {}

for key, value in checkpoint["state_dict"].items():

    if key.startswith("model."):

        state_dict[key.replace("model.", "", 1)] = value

# Load weights
missing_keys, unexpected_keys = model.load_state_dict(
    state_dict,
    strict=True
)

print("Model loaded successfully!")

model.to(device)
model.eval()

# ======================================================
# IMAGE TRANSFORM
# ======================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ======================================================
# PREDICT ONE IMAGE
# ======================================================

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)

        probs = torch.softmax(output, dim=1)[0]

        pred = torch.argmax(probs).item()

    return (
        pred,
        probs[0].item(),
        probs[1].item()
    )

# ======================================================
# EVALUATE TEST SET
# ======================================================

results = []

y_true = []
y_pred = []

label_dict = {
    "real":0,
    "fake":1
}

for folder in ["real","fake"]:

    folder_path = os.path.join(TEST_FOLDER, folder)

    print(f"\nProcessing {folder} images...")

    for filename in sorted(os.listdir(folder_path)):

        if not filename.lower().endswith((".jpg",".jpeg",".png")):
            continue

        image_path = os.path.join(folder_path, filename)

        pred, real_prob, fake_prob = predict(image_path)

        actual = label_dict[folder]

        y_true.append(actual)
        y_pred.append(pred)

        results.append({

            "Filename": filename,

            "Actual Label":
                "REAL" if actual == 0 else "FAKE",

            "Predicted Label":
                "REAL" if pred == 0 else "FAKE",

            "Real Probability":
                round(real_prob,4),

            "Fake Probability":
                round(fake_prob,4),

            "Correct":
                actual == pred

        })

# ======================================================
# CALCULATE METRICS
# ======================================================

accuracy = accuracy_score(y_true,y_pred)

precision = precision_score(y_true,y_pred)

recall = recall_score(y_true,y_pred)

f1 = f1_score(y_true,y_pred)

cm = confusion_matrix(y_true,y_pred)

metrics = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1
    ]

})

confusion = pd.DataFrame(

    cm,

    index=[
        "Actual REAL",
        "Actual FAKE"
    ],

    columns=[
        "Predicted REAL",
        "Predicted FAKE"
    ]

)

# ======================================================
# SAVE TO EXCEL
# ======================================================

with pd.ExcelWriter(OUTPUT_FILE) as writer:

    pd.DataFrame(results).to_excel(
        writer,
        sheet_name="Predictions",
        index=False
    )

    metrics.to_excel(
        writer,
        sheet_name="Metrics",
        index=False,
        startrow=0
    )

    confusion.to_excel(
        writer,
        sheet_name="Metrics",
        startrow=8
    )

print("\n========================================")
print("Evaluation Complete")
print("========================================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"\nResults saved to {OUTPUT_FILE}")