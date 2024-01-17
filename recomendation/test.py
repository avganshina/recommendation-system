import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import tensorflow_datasets as tfds
from persistance.logger import Logger
from core.constants import COLAB_MODEL_PATH, VECTORIZER_PATH, SCALER_PATH, HYBRID_MODEL_PATH

def load_and_preprocess_data(datapath):
    # Load and preprocess ratings data
    ratings_data = tfds.load(datapath, split='train')
    df = tfds.as_dataframe(ratings_data)
    df = df[:10]  # Just using what the laptop can handle
    return df

def transform_df(df):
    dff = {}
    for i in range(len(df['movie_genres'].keys())):
        dff[str(i)] = df['movie_genres'][i].astype(str)
    return dff

if __name__ == "__main__":
    # Set up logger
    logger = Logger()

    # Test the recommendation system
    logger.log("started testing recommendation system")
    collab_model = joblib.load(COLAB_MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    scaler = joblib.load(SCALER_PATH)
    hybrid_model = joblib.load(HYBRID_MODEL_PATH)
    # Load and preprocess test data
    test_data = load_and_preprocess_data('movielens/latest-small-ratings')

    # Extract actual ratings
    actual_ratings = test_data['user_rating'].values

    # Extract collaborative filtering features
    collab_features = [collab_model.predict(row['user_id'], row['movie_id']).est for _, row in test_data.iterrows()]

    # TF-IDF Vectorization and Scaling for content-based features
    tfidf_features = tfidf_vectorizer.transform(transform_df(test_data))
    tfidf_features_scaled = scaler.transform(tfidf_features.toarray())

    # Combine collaborative filtering and TF-IDF features
    hybrid_features = np.concatenate([np.array(collab_features).reshape(-1, 1), tfidf_features_scaled], axis=1)

    # Make predictions using the loaded hybrid model
    predicted_ratings = hybrid_model.predict(hybrid_features)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(actual_ratings, predicted_ratings)

    logger.log(f"Mean Squared Error: {mse}")
    logger.log("Finished testing recommendation system")
