# SMS Spam Classifier

A complete beginner-friendly SMS spam classification project. It trains a text classifier on the public UCI SMS Spam Collection dataset, compares simple models, tunes a decision threshold, and serves predictions in a small Flask web app.

## What This Project Does

The app takes an SMS message as input and returns:

- `spam` or `ham` prediction
- Confidence score
- Raw spam probability
- Threshold used by the final model
- Model type selected during training

## Files In This Project

- `train_model.py`: Downloads the dataset, cleans SMS text, trains and compares models, tunes a threshold, prints metrics, and saves the final model.
- `app.py`: Flask app with a browser form at `/` and a JSON prediction endpoint at `/predict`.
- `requirements.txt`: Python packages needed for the machine-learning and Flask app.
- `src/App.tsx`: React/Vite visual demo page for the project.
- `README.md`: This start guide.

## Dataset

Use the SMS Spam Collection dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The training script downloads this zip file automatically:

https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip

The dataset contains SMS messages labeled as:

- `ham`: legitimate message
- `spam`: unwanted promotional, phishing, or scam message

## How To Start

1. Create and activate a Python virtual environment.

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Train the model.

```bash
python train_model.py
```

This command will:

- Download the UCI dataset
- Print spam vs ham counts
- Clean messages with lowercase text, URL normalization, symbol cleanup, and whitespace cleanup
- Convert text to TF-IDF features
- Train Naive Bayes
- Train Logistic Regression with balanced class weights
- Train a calibrated Linear SVM with balanced class weights
- Tune the spam threshold to reduce false negatives
- Print precision, recall, F1, and confusion matrix
- Save the best model to `artifacts/sms_spam_model.joblib`

4. Run the Flask app.

```bash
python app.py
```

5. Open the app in your browser.

```text
http://127.0.0.1:5000
```

## JSON API Example

After starting Flask, send a POST request to `/predict`.

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"URGENT! You won a free prize. Click now to claim."}'
```

Example response:

```json
{
  "label": "spam",
  "confidence": 0.9821,
  "spam_probability": 0.9821,
  "threshold": 0.43,
  "model": "linear_svm"
}
```

## Machine Learning Workflow

1. Load the SMS Spam Collection dataset.
2. Check label counts to understand class imbalance.
3. Clean message text.
4. Convert text to numeric features with `TfidfVectorizer`.
5. Train `MultinomialNB` as a strong baseline.
6. Train `LogisticRegression` with `class_weight="balanced"`.
7. Train calibrated `LinearSVC` with `class_weight="balanced"`.
8. Tune the probability threshold on a validation set.
9. Evaluate on a held-out test set with precision, recall, F1, and confusion matrix.
10. Save the best model and threshold with `joblib`.

## Why Threshold Tuning Matters

Spam detection has an important tradeoff:

- False positive: a real message is incorrectly marked as spam.
- False negative: a spam or phishing message reaches the inbox.

For this project, false negatives are treated as more costly, so the threshold tuning step prioritizes high spam recall while still checking precision and F1.

## Run The React Demo Page

The React page is a polished front-end concept for the classifier workflow. It includes a lightweight in-browser demo so you can show the product idea even before connecting a live API.

```bash
npm run dev
```

Then open the local Vite URL shown in the terminal.

## Next Improvements

- Connect the React UI directly to the Flask `/predict` endpoint.
- Add model versioning and a saved metrics report.
- Add more phishing-specific features such as URL domain reputation.
- Log uncertain predictions for human review.
- Deploy Flask with Gunicorn on Render, Railway, or another Python-friendly host.

## Author

- Ishan Maitra - AI Developer
- Contact no. - +91 9674026774
- Email id - ishanmaitra2012@gamil.com
- Github - https://github.com/maitrawebtech/
