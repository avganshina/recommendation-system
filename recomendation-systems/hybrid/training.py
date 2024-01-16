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

ratings_data= tfds.load('movielens/latest-small-ratings', split='train')
df = tfds.as_dataframe(ratings_data)
df = df[:int(len(df) * 0.001)]
selected_columns = df[['user_id', 'movie_id', 'movie_genres', 'user_rating']]

# Collaborative Filtering Model
print('started with CFM')
collab_model = SVD()
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'user_rating']], reader)
trainset = data.build_full_trainset()
collab_model.fit(trainset)
joblib.dump(collab_model, 'models/collab_model.joblib')
print('Saved CFM')

# TF-IDF Vectorization and Scaling
print('started with CBM')
tfidf_vectorizer = TfidfVectorizer()

dff = {}
for i in range(len(df['movie_genres'].keys())):
    dff[str(i)] = df['movie_genres'][i].astype(str)
    
tfidf_vectorizer = tfidf_vectorizer.fit(dff)
print('Fitted Vectorizer')
tfidf_features = tfidf_vectorizer.transform(dff)
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
scaler = StandardScaler()
scaler= scaler.fit(tfidf_features.toarray())
print('Fitted Scaler')
joblib.dump(scaler, 'models/scaler.joblib')
tfidf_features_scaled = scaler.transform(tfidf_features.toarray())
print('finished with CBM')

# Combine collaborative filtering and TF-IDF features
collab_features = [collab_model.predict(row['user_id'], row['movie_id']).est for _, row in df.iterrows()]
hybrid_features = np.concatenate([np.array(collab_features).reshape(-1, 1), tfidf_features_scaled], axis=1)

# Target variable
target_variable = df['user_rating']

# Define the hybrid model pipeline
hybrid_model = Pipeline([
    ('regressor', LinearRegression())  # You can replace LinearRegression with your desired regression model
])

# Fit the hybrid model with the combined features and target variable
print('started with Hybrid')
hybrid_model.fit(hybrid_features, target_variable)
print('finished with Hybrid')
joblib.dump(hybrid_model, 'models/hybrid_model.joblib')
