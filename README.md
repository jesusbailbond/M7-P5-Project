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
