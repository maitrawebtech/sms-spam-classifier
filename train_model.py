from __future__ import annotations

import argparse
import io
import math
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "sms_spam_model.joblib"


def clean_text(text: str) -> str:
    """Lowercase and keep useful SMS tokens while removing noisy symbols."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"[^a-z0-9$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_sms_collection(url: str = DATA_URL) -> pd.DataFrame:
    with urlopen(url) as response:
        zipped = zipfile.ZipFile(io.BytesIO(response.read()))
        with zipped.open("SMSSpamCollection") as dataset_file:
            df = pd.read_csv(dataset_file, sep="\t", names=["label", "message"], encoding="latin-1")

    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_message"] = df["message"].map(clean_text)
    return df


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    strip_accents="unicode",
                ),
            ),
            ("model", model),
        ]
    )


def predict_spam_probability(pipeline: Pipeline, messages: pd.Series) -> pd.Series:
    if hasattr(pipeline, "predict_proba"):
        return pd.Series(pipeline.predict_proba(messages)[:, 1], index=messages.index)

    scores = pipeline.decision_function(messages)
    probabilities = [1 / (1 + math.exp(-score)) for score in scores]
    return pd.Series(probabilities, index=messages.index)


def tune_threshold(y_true: pd.Series, spam_probability: pd.Series) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for threshold in [x / 100 for x in range(20, 81)]:
        predictions = (spam_probability >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predictions, pos_label=1, average="binary", zero_division=0
        )
        # Missing spam is costly, so recall is prioritized while still keeping precision usable.
        if recall >= 0.90 and f1 > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = {"precision": precision, "recall": recall, "f1": f1}

    return best_threshold, best_metrics


def train(output_path: Path = MODEL_PATH) -> None:
    df = load_sms_collection()
    print("Class counts:")
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_num"]
    )
    train_df, validation_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["label_num"]
    )

    candidates = {
        "naive_bayes": make_pipeline(MultinomialNB(alpha=0.2)),
        "logistic_regression": make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear")
        ),
        "linear_svm": make_pipeline(
            CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42), cv=3)
        ),
    }

    results = []
    for name, pipeline in candidates.items():
        pipeline.fit(train_df["clean_message"], train_df["label_num"])
        validation_probability = predict_spam_probability(pipeline, validation_df["clean_message"])
        threshold, threshold_metrics = tune_threshold(validation_df["label_num"], validation_probability)
        validation_prediction = (validation_probability >= threshold).astype(int)
        f1 = f1_score(validation_df["label_num"], validation_prediction)
        results.append((f1, name, pipeline, threshold, threshold_metrics))
        print(f"\n{name} validation report at threshold {threshold:.2f}:")
        print(classification_report(validation_df["label_num"], validation_prediction, target_names=["ham", "spam"]))

    _, best_name, best_pipeline, best_threshold, best_validation_metrics = max(results, key=lambda item: item[0])
    test_probability = predict_spam_probability(best_pipeline, test_df["clean_message"])
    test_prediction = (test_probability >= best_threshold).astype(int)

    print(f"\nSelected model: {best_name}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Validation threshold metrics: {best_validation_metrics}")
    print("\nFinal test report:")
    print(classification_report(test_df["label_num"], test_prediction, target_names=["ham", "spam"]))
    print("Confusion matrix [[ham correct, ham as spam], [spam as ham, spam correct]]:")
    print(confusion_matrix(test_df["label_num"], test_prediction))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "threshold": best_threshold,
            "model_name": best_name,
        },
        output_path,
    )
    print(f"\nSaved model artifact to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an SMS spam classifier on the UCI dataset.")
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help="Where to save the joblib model artifact.")
    args = parser.parse_args()
    train(args.output)