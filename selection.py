"""Selection script
Replace this file with your own logic"""
import pandas as pd
import numpy as np
import shutil


class Predictor:
    def __init__(self, embeddings_path: str):
        print("Loading embeddings")
        self.embeddings = self.read_parquet(embeddings_path)
        self.embeddings_vect = np.stack(self.embeddings["embedding"].to_numpy())
        print("Embeddings loaded")

    def read_parquet(self, path: str):
        return pd.read_parquet(path, engine="pyarrow", use_threads=True)

    def similarity(self, centroid, n_closest, n_random):
        random_indexes = np.random.choice(
            len(self.embeddings_vect), size=n_random, replace=False
        )
        distances = np.linalg.norm(
            self.embeddings_vect[random_indexes] - centroid, axis=1
        )
        result = np.argsort(distances)[:n_closest]
        return result

    def get_embeddings_labeled_data(self, input_path):
        df = pd.read_csv(input_path)
        return self.embeddings.loc[self.embeddings["ImageID"].isin(df["ImageID"])]

    def calculate_centroid(self, df):
        return df["embedding"].apply(lambda x: np.array(x)).values.mean()

    def closest_and_furthest(
        self, input_path, output_path, n_closest=100, n_random=1000
    ):
        df = self.get_embeddings_labeled_data(input_path)
        print("Embeddings loaded")
        print("Getting centroids")
        centroid = self.calculate_centroid(df)
        print("Running similairty")
        similarity = self.similarity(centroid, n_closest, n_random)
        print("Saving submission")
        submission = pd.DataFrame(self.embeddings.iloc[similarity]["ImageID"])
        submission["Confidence"] = 1
        submission.to_csv(output_path, index=False)
