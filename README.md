# Mood-Journal-Analyzer

This project is a text classification model that predicts the mood of a journal entry as one of four categories: sad, calm, excited, or anxious. The model uses a TF-IDF vectorizer to transform text data and a Logistic Regression classifier to predict the mood. Additionally, SHAP (SHapley Additive exPlanations) is used to visualize which words had the most influence on the model's predictions.

The repository contains the following files:

Mood_Journal_Data_Expanded.csv – The dataset of journal entries labeled with moods.
Mood_Journal_Model.ipynb – The Python notebook containing the code for training, evaluating, and explaining the model (created in Google Colab).
mood_journal_model.joblib – The pre-trained model saved for reuse.
shap_summary_plot.png – SHAP summary plot showing which words influenced each mood class the most.

## How It Works:
- The journal entries are preprocessed using a TF-IDF vectorizer, which converts the text into numerical features.
- A Logistic Regression model is trained on these features to classify the entries into moods.
- Cross-validation is used to ensure that the model generalizes well to unseen data.
- SHAP explains which words were most impactful for the model's predictions.

## Setup Instructions:
1. Clone the repository
2. Install required Python libraries (Python 3.8+ recommended):
3. pip install pandas numpy scikit-learn shap joblib matplotlib
4. Run the notebook:
Open Mood_Journal_Model.ipynb in Jupyter Notebook or JupyterLab and run the cells to train the model, evaluate its performance, and generate the SHAP plot.
If you want to use the pre-trained model instead of retraining, load it with:
import joblib
model = joblib.load("mood_journal_model.joblib")

## Results
Cross-Validation Accuracy: The model is evaluated using 5-fold stratified cross-validation to account for class balance. The notebook prints:
- Accuracy per fold
- Mean accuracy
- Standard deviation

Feature Importance (SHAP):
The SHAP summary plot identifies which words contributed the most to predicting each mood category, providing transparency into the model's decisions.
