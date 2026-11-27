# Toxic Comment Detection and Moderation System Using NLP

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended):
   - Windows: `python -m venv .venv`
   - Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
4. Install dependencies: `pip install -r requirements.txt`
5. Train the model and create evaluation outputs: `python train_model.py`
6. Launch the Streamlit application: `streamlit run app.py`

The browser will open the local app (usually at `http://localhost:8501`).

## Files

- `app.py`: Streamlit application.
- `train_model.py`: training, evaluation, model saving and confusion matrix generation.
- `dataset/comments.csv`: labelled demonstration dataset.
- `models/`: trained model is created after training.
- `reports/`: metrics, confusion matrix and held-out predictions are created after training.
- `Project_Report.docx` / `.pdf`: project documentation.

## Method

A train/test split is made before fitting vectorizers. A scikit-learn Pipeline contains word-level TF-IDF (unigrams and bigrams), character-level TF-IDF (3–5 character n-grams), and Logistic Regression. The vectorizers are fitted only on training data to avoid test-set vocabulary leakage.

The Streamlit threshold is applied to the model's Toxic class score. The app flags content for human review; it does not automatically delete comments or punish users.

## Dataset and limitations

The included dataset is synthetic, authored for demonstration and code testing. It is not a validated toxicity corpus, does not represent all dialects or communities, and is too small to establish real-world accuracy. Metrics produced from it are demonstration metrics only. Replace it with a sufficiently large, representative, licensed and human-annotated dataset before any real deployment. Inspect false positives and false negatives and evaluate subgroup performance.

## Deploying online

For a public live application, deploy the project to a Streamlit-compatible hosting service. Include `requirements.txt`, `app.py`, `train_model.py`, and `dataset/comments.csv`; ensure the training step has been run or configure the host to run `python train_model.py` before `streamlit run app.py`. Do not upload private or sensitive comments.
