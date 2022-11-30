import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA
from pandera.typing import Series, DataFrame

import ast

from embeddings_dimension_reduction import StandardScaler

def extract_embeddings_features(embeddings: Series):
	"""
	Args:
		embeddings (np.ndarray): array of string representation of feature vectors.
	Return:
		(np.ndarray): feature matrix of dimension (n_observations, n_features)
	"""
	n = embeddings.shape[0]
	embeddings_matrix = []

	for i in range(n):
		# embeddings are encoded as string representation of vector
		# convert these into list
		feature_vector = ast.literal_eval(embeddings.iloc[i])
		print(feature_vector)
		embeddings_matrix.append(feature_vector)

	print(embeddings_matrix)
	return embeddings_matrix

def reduce_embeddings_dimension(embeddings_array: list, prefix: str, total_variance_explained: float = 0.95) -> DataFrame: 
	print(f"reducing embeddings dimension with PCA keeping {total_variance_explained * 100}% of variance explained...")
	pca = PCA(n_components=total_variance_explained)

	scaler = StandardScaler()
	scaler.fit_transform()

	embeddings_reduced_array = pca.fit_transform(embeddings_array)
	print(f"successfully reduced from {len(embeddings_array[0])} features to {embeddings_reduced_array.shape[1]} features")

	embeddings_reduced_df = pd.DataFrame(embeddings_reduced, index = embeddings_df.index)
	embeddings_reduced_df = embeddings_reduced_df.add_prefix(prefix)

	return embeddings_reduced_array

def array_to_df(embeddings_array):
	columns = []

	# create column names
	columns.extend(["img_embeddings" + str(i + 1) for i in range(len(first_embedding))])

	# create embeddings features pandas dataframe
	embeddings_df = pd.Series(embeddings_array).apply(pd.Series)

	# naming columns and index
	embeddings_df = pd.DataFrame(embeddings_df, index=embeddings.index)
	embeddings_df.columns = columns

	return embeddings_df

	