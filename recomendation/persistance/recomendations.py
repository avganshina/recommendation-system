import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow_datasets as tfds
import joblib

class CollaborativeFilteringModel:
    def __init__(self):
        self.model = SVD()
        self.reader = Reader(rating_scale=(1, 5))

    def fit(self, data):
        dataset = Dataset.load_from_df(data[['user_id', 'movie_id', 'user_rating']], self.reader)
        trainset = dataset.build_full_trainset()
        self.model.fit(trainset)

    def save_model(self, path):
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        return joblib.load(path)

    def predict(self, user_id, movie_id):
        return self.model.predict(user_id, movie_id).est

class ContentBasedModel:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.scaler = StandardScaler()

    def fit_transform(self, data):
        self.tfidf_vectorizer.fit(data)
        tfidf_features = self.tfidf_vectorizer.transform(data)

        self.scaler.fit(tfidf_features.toarray())
        tfidf_features_scaled = self.scaler.transform(tfidf_features.toarray())

        return tfidf_features_scaled

    def save_vectorizer(self, path):
        joblib.dump(self.tfidf_vectorizer, path)

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

    def load_vectorizer(self, path):
        self.tfidf_vectorizer = joblib.load(path)

    def load_scaler(self, path):
        self.scaler = joblib.load(path)

    def transform(self, data):
        tfidf_features = self.tfidf_vectorizer.transform(data)
        tfidf_features_scaled = self.scaler.transform(tfidf_features.toarray())
        return tfidf_features_scaled

class HybridModel:
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, features, target):
        self.regressor.fit(features, target)

    def save_model(self, path):
        joblib.dump(self.regressor, path)

    def load_model(self, path):
        self.regressor = joblib.load(path)

    def predict(self, features):
        return self.regressor.predict(features)