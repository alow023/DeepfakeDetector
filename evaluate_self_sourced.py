import os
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_model.ckpt"

TEST_FOLDER = r"C:\arissa\Deepfake detection\Data\self-sourced"

OUTPUT_FILE = "prediction_results (Self sourced).xlsx"

VISUALIZATION_FOLDER = "results_visualisations (Self sourced)"

os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("Loading checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = efficientnet_b0()

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        2
    )

    # Lightning saves parameters as model.xxx
    state_dict = {}

    for key, value in checkpoint["state_dict"].items():

        if key.startswith("model."):

            new_key = key.replace("model.", "", 1)

            state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("\nModel Loaded")

    if len(missing) > 0:
        print("Missing keys:")
        print(missing)

    if len(unexpected) > 0:
        print("Unexpected keys:")
        print(unexpected)

    model.to(DEVICE)

    model.eval()

    return model


model = load_model()

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])

# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probs = torch.softmax(output, dim=1)[0]

        prediction = torch.argmax(probs).item()

    return (

        prediction,

        probs[0].item(),

        probs[1].item()

    )

# ============================================================
# EVALUATE DATASET
# ============================================================

results = []

y_true = []
y_pred = []

summary = {}

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

folders = sorted([
    f for f in os.listdir(TEST_FOLDER)
    if os.path.isdir(os.path.join(TEST_FOLDER, f))
])

print("\n===================================")
print("Starting Evaluation")
print("===================================")

for folder in folders:

    folder_path = os.path.join(TEST_FOLDER, folder)

    print(f"\nProcessing folder: {folder}")

    # Only Real folder is considered REAL
    actual = 0 if folder.lower() == "real" else 1

    summary[folder] = {

        "Images": 0,

        "Predicted Real": 0,

        "Predicted Fake": 0,

        "Correct": 0

    }

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(VALID_EXTENSIONS)
    ])

    print(f"Found {len(image_files)} images")

    for i, filename in enumerate(image_files, start=1):

        image_path = os.path.join(folder_path, filename)

        try:

            pred, real_prob, fake_prob = predict(image_path)

        except Exception as e:

            print(f"Skipping {filename}: {e}")

            continue

        y_true.append(actual)
        y_pred.append(pred)

        summary[folder]["Images"] += 1

        if pred == 0:
            summary[folder]["Predicted Real"] += 1
        else:
            summary[folder]["Predicted Fake"] += 1

        correct = (pred == actual)

        if correct:
            summary[folder]["Correct"] += 1

        confidence = max(real_prob, fake_prob)

        results.append({

            "Generator": folder,

            "Filename": filename,

            "Ground Truth":
                "REAL" if actual == 0 else "FAKE",

            "Prediction":
                "REAL" if pred == 0 else "FAKE",

            "Confidence":
                round(confidence,4),

            "Real Probability":
                round(real_prob,4),

            "Fake Probability":
                round(fake_prob,4),

            "Correct":
                correct

        })

        print(
            f"[{i}/{len(image_files)}] "
            f"{filename:<25} "
            f"Pred: {'REAL' if pred==0 else 'FAKE'} "
            f"({confidence:.4f})"
        )

print("\nFinished evaluating all folders.")

# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred
)

metrics_df = pd.DataFrame({

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

confusion_df = pd.DataFrame(

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

# ============================================================
# GENERATOR SUMMARY
# ============================================================

summary_rows = []

for generator, stats in summary.items():

    total = stats["Images"]

    detection_rate = (
        stats["Correct"] / total
        if total > 0 else 0
    )

    summary_rows.append({

        "Generator": generator,

        "Images": total,

        "Predicted REAL": stats["Predicted Real"],

        "Predicted FAKE": stats["Predicted Fake"],

        "Correct": stats["Correct"],

        "Detection Rate (%)":
            round(detection_rate * 100, 2)

    })

summary_df = pd.DataFrame(summary_rows)

# Create dataframe BEFORE plotting
predictions_df = pd.DataFrame(results)

# ============================================================
# VISUALISATIONS
# ============================================================

print("\nGenerating visualisations...")

# -----------------------------
# 1. Confusion Matrix
# -----------------------------

plt.figure(figsize=(6,5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.xticks([0,1], ["REAL","FAKE"])

plt.yticks([0,1], ["REAL","FAKE"])

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i,j],
            ha="center",
            va="center",
            fontsize=12,
            color="black"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(VISUALIZATION_FOLDER, "confusion_matrix.png"),
    dpi=300
)

plt.close()


# -----------------------------
# 2. Detection Rate by Generator
# -----------------------------

plt.figure(figsize=(9,5))

plt.bar(
    summary_df["Generator"],
    summary_df["Detection Rate (%)"]
)

plt.ylabel("Detection Rate (%)")

plt.xlabel("Generator")

plt.title("Detection Accuracy by Generator")

plt.xticks(rotation=45, ha="right")

plt.ylim(0,100)

plt.tight_layout()

plt.savefig(
    os.path.join(VISUALIZATION_FOLDER, "generator_detection_rate.png"),
    dpi=300
)

plt.close()


# -----------------------------
# 3. Confidence Distribution
# -----------------------------

real_scores = predictions_df[
    predictions_df["Ground Truth"]=="REAL"
]["Confidence"]

fake_scores = predictions_df[
    predictions_df["Ground Truth"]=="FAKE"
]["Confidence"]

plt.figure(figsize=(8,5))

plt.hist(
    real_scores,
    bins=20,
    alpha=0.6,
    label="REAL"
)

plt.hist(
    fake_scores,
    bins=20,
    alpha=0.6,
    label="FAKE"
)

plt.xlabel("Prediction Confidence")

plt.ylabel("Number of Images")

plt.title("Confidence Distribution")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(VISUALIZATION_FOLDER, "confidence_distribution.png"),
    dpi=300
)

plt.close()

print("Visualisations saved.")

# ============================================================
# SAVE TO EXCEL
# ============================================================

print("\nSaving results to Excel...")

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    # --------------------------------------------
    # Sheet 1 - Individual Predictions
    # --------------------------------------------
    predictions_df.to_excel(
        writer,
        sheet_name="Predictions",
        index=False
    )

    # --------------------------------------------
    # Sheet 2 - Generator Summary
    # --------------------------------------------
    summary_df.to_excel(
        writer,
        sheet_name="Generator Summary",
        index=False
    )

    # --------------------------------------------
    # Sheet 3 - Overall Metrics
    # --------------------------------------------
    metrics_df.to_excel(
        writer,
        sheet_name="Metrics",
        startrow=0,
        index=False
    )

    confusion_df.to_excel(
        writer,
        sheet_name="Metrics",
        startrow=8
    )

print("\nExcel file saved successfully!")

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"Total Images : {len(results)}")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")

print("\nConfusion Matrix")
print(confusion_df)

print("\n")
print("=" * 70)
print("PER GENERATOR RESULTS")
print("=" * 70)

for _, row in summary_df.iterrows():

    print(
        f"{row['Generator']:<20}"
        f"Images: {int(row['Images']):>4}   "
        f"Correct: {int(row['Correct']):>4}   "
        f"Detection Rate: {row['Detection Rate (%)']:>6.2f}%"
    )

print("\n")
print("=" * 70)
print(f"Results written to: {OUTPUT_FILE}")
print(f"Visualisations written to: {VISUALIZATION_FOLDER}")
print("=" * 70)