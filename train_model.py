"""
Train and evaluate the toxic-comment moderation baseline.
Run from the project root: python train_model.py
The bundled dataset is synthetic demonstration data, not a production benchmark.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "comments.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna(subset=["comment", "label"])
df["comment"] = df["comment"].astype(str)
df["label"] = df["label"].astype(str).str.strip()

required = {"Safe", "Toxic"}
if set(df["label"].unique()) != required:
    raise ValueError("Dataset must contain exactly the labels 'Safe' and 'Toxic'.")

X_train, X_test, y_train, y_test = train_test_split(
    df["comment"], df["label"],
    test_size=0.20, random_state=42, stratify=df["label"]
)

# Both vectorizers are fitted inside the pipeline on training data only.
features = FeatureUnion([
    ("word", TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        ngram_range=(1, 2), min_df=1, sublinear_tf=True
    )),
    ("char", TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5),
        min_df=1, sublinear_tf=True
    ))
])

model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
classes = list(model.named_steps["classifier"].classes_)
toxic_index = classes.index("Toxic")

accuracy = accuracy_score(y_test, pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, pred, labels=["Safe", "Toxic"], average="binary", pos_label="Toxic", zero_division=0
)

joblib.dump(model, MODEL_DIR / "toxic_comment_model.joblib")

metrics = {
    "dataset_rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "accuracy": float(accuracy),
    "toxic_precision": float(precision),
    "toxic_recall": float(recall),
    "toxic_f1": float(f1),
    "classification_report": classification_report(
        y_test, pred, output_dict=True, zero_division=0
    ),
    "confusion_matrix_labels": ["Safe", "Toxic"],
    "confusion_matrix": confusion_matrix(
        y_test, pred, labels=["Safe", "Toxic"]
    ).tolist(),
    "note": "Metrics are from synthetic demonstration data and must not be treated as real-world performance."
}
(REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, labels=["Safe", "Toxic"],
    display_labels=["Safe", "Toxic"], cmap="Blues", ax=ax, colorbar=False
)
ax.set_title("Confusion Matrix - Synthetic Demo Dataset")
fig.tight_layout()
fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
plt.close(fig)

# Save held-out examples and predictions for inspection/error analysis.
errors = pd.DataFrame({
    "comment": X_test.reset_index(drop=True),
    "actual": y_test.reset_index(drop=True),
    "predicted": pred
})
errors["correct"] = errors["actual"] == errors["predicted"]
errors.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

print("Training complete.")
print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Accuracy: {accuracy:.3f} | Toxic precision: {precision:.3f} | "
      f"recall: {recall:.3f} | F1: {f1:.3f}")
print("\n", classification_report(y_test, pred, zero_division=0))
print(f"Model saved to: {MODEL_DIR / 'toxic_comment_model.joblib'}")
