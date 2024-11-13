Reddit Data Analysis and Topic Clustering with Random Forest and KMeans
This project collects, analyzes, and clusters data from the Reddit subreddit throneandliberty. Using natural language processing (NLP) and machine learning techniques, it classifies post scores and identifies important textual features. Key steps include data collection, preprocessing, classification, clustering, and feature importance analysis.

Project Structure
Data Collection: Using the PRAW library, top submissions from the throneandliberty subreddit are collected.
Text Preprocessing: Text data is preprocessed to remove punctuation, convert to lowercase, and remove stopwords.
Feature Extraction: TF-IDF vectorization is used to create a feature matrix from the cleaned post bodies.
Binary Classification: A RandomForestClassifier is trained to classify posts as high-score or low-score based on a score threshold.
Hyperparameter Tuning: RandomizedSearchCV is applied to find optimal Random Forest parameters.
Clustering: KMeans clustering groups posts into clusters to uncover latent topics.
Feature Importance Analysis: Feature importances from the trained model are visualized.
Confusion Matrix Visualization: Model predictions are evaluated using a confusion matrix heatmap.
Cross-Validation: Cross-validation scores for model robustness are calculated.
New Data Prediction: Predictions are made on new example data based on the trained model.
Requirements
The project requires the following Python libraries:

pandas
praw
datetime
numpy
string
nltk
matplotlib
seaborn
scikit-learn
Install packages using:

bash
Copier le code
pip install praw numpy pandas matplotlib seaborn scikit-learn nltk
Installation and Setup
Clone this repository:
bash
Copier le code
git clone <repository-url>
cd <repository-name>
Install dependencies using the requirements.txt file.
Set up your Reddit API credentials and enter them in the praw.Reddit section of the code.
Instructions
Run Data Collection: The script collects data from the subreddit, extracting titles, scores, IDs, URLs, and body text.

Preprocess Text: The preprocess_text function cleans text data, removing punctuation and stopwords.

TF-IDF Vectorization: A TF-IDF vectorizer is used to convert the cleaned text into numerical features for classification.

Classification and Clustering:

Train a RandomForestClassifier to classify high- vs. low-scoring posts.
Hyperparameters are optimized with RandomizedSearchCV.
Perform topic clustering with KMeans.
Evaluation and Visualization:

Plot feature importances and visualize the confusion matrix using seaborn heatmaps.
Display cross-validation scores for model evaluation.
Predict on New Data: Predict labels for new example data using the trained model.

Example Usage
python
Copier le code
# Example new data for prediction
new_data = ['I really hate this game', 'The loot we get as F2P in dungeon, archboss or field boss cant compare even if we get lucent']

# Preprocess and vectorize new data
new_data_clean = [preprocess_text(text) for text in new_data]
new_data_vectorized = vectorizer.transform(new_data_clean)

# Predict using the best model
new_predictions = best_rf_model.predict(new_data_vectorized)
print(new_predictions)
Results
Classification Report: Accuracy, precision, recall, and F1-scores for high- and low-scoring posts.
Feature Importances: Displays the top 20 features contributing to high- and low-score classification.
Confusion Matrix: A heatmap showing true vs. predicted labels.
Cross-Validation Scores: The model's performance over 5-fold cross-validation.
