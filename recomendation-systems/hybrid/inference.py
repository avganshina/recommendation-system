from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from surprise import SVD, Dataset, Reader
from surprise.dump import load
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import joblib

tfidf_vectorizer = TfidfVectorizer()

# Making up some data
inference_data = pd.DataFrame({
    'user_id': [4, 4, 4],
    'movie_id': [404, 150, 606],
    'movie_genres': ['0', '9', '10']
})

hybrid_model = joblib.load('models/hybrid_model.joblib')
collab_model = joblib.load('models/collab_model.joblib')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
scaler = joblib.load('models/scaler.joblib')

# Extract collaborative filtering features
collab_features = [collab_model.predict(row['user_id'], row['movie_id']).est for _, row in inference_data.iterrows()]

# TF-IDF Vectorization and Scaling for content-based features
tfidf_features = tfidf_vectorizer.transform(inference_data['movie_genres'])
tfidf_features_scaled = scaler.transform(tfidf_features.toarray())

# Combine collaborative filtering and TF-IDF features
hybrid_features = np.concatenate([np.array(collab_features).reshape(-1, 1), tfidf_features_scaled], axis=1)

# Make predictions using the loaded hybrid model
predictions = hybrid_model.predict(hybrid_features)

# Extract predicted ratings
predicted_ratings = predictions

print('Predicted Ratings:', predicted_ratings)