import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow_datasets as tfds
from persistance.recomendations import HybridModel, CollaborativeFilteringModel, ContentBasedModel
from persistance.logger import Logger
from core.constants import COLAB_MODEL_PATH, VECTORIZER_PATH, SCALER_PATH, HYBRID_MODEL_PATH

def __transform_df(df):
    dff = {}
    for i in range(len(df['movie_genres'].keys())):
        dff[str(i)] = df['movie_genres'][i].astype(str)
    return dff

if __name__ == "__main__":

    # Load and preprocess ratings data
    ratings_data = tfds.load('movielens/latest-small-ratings', split='train')
    df = tfds.as_dataframe(ratings_data)
    df = df[:int(len(df) * 0.001)] # Just using what laptop can handle

    # Initialize logger
    logger = Logger()

    # Collaborative Filtering Model
    collab_model = CollaborativeFilteringModel()
    logger.log(f'started training collaborative filtering model')
    collab_model.fit(df)
    collab_model.save_model(COLAB_MODEL_PATH)
    logger.log(f'finished training collaborative filtering model')

    # Content-Based Model
    content_model = ContentBasedModel()
    logger.log(f'started training content based model')
    dff = __transform_df(df)
    tfidf_features_scaled = content_model.fit_transform(dff)
    content_model.save_vectorizer(VECTORIZER_PATH)
    content_model.save_scaler(SCALER_PATH)
    logger.log(f'finished training content based model')

    # Combine collaborative filtering and TF-IDF features
    collab_features = [collab_model.model.predict(row['user_id'], row['movie_id']).est for _, row in df.iterrows()]
    hybrid_features = np.concatenate([np.array(collab_features).reshape(-1, 1), tfidf_features_scaled], axis=1)

    # Target variable
    target_variable = df['user_rating']

    # Define the hybrid model pipeline
    hybrid_regressor = LinearRegression()
    hybrid_model = HybridModel(hybrid_regressor)

    # Fit the hybrid model with the combined features and target variable
    logger.log(f'started training hybrid model')
    hybrid_model.fit(hybrid_features, target_variable)
    hybrid_model.save_model(HYBRID_MODEL_PATH)
    logger.log(f'finished training hybrid model')