import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA
from pandera.typing import Series, DataFrame

import ast

def extract_embeddings_features(embeddings: Series[str]) -> DataFrame:
	columns = []
	embeddings_list = []

	# embeddings are encoded as string representation of vector
	# convert these into list
	for i in range(embeddings.shape[0]):
		embeddings_list.append(ast.literal_eval(embeddings.iloc[i]))

	first_embedding = embeddings_list[0]

	# create column names
	columns.extend(["img_embeddings" + str(i + 1) for i in range(len(first_embedding))])

	# create embeddings features pandas dataframe
	embeddings_df = pd.Series(embeddings_list).apply(pd.Series)

	# naming columns and index
	embeddings_df = pd.DataFrame(embeddings_df, index=embeddings.index)
	embeddings_df.columns = columns

	return embeddings_df

def reduce_embeddings_dimension(embeddings_df: DataFrame, prefix: str, total_variance_explained: float = 0.95) -> DataFrame: 
	print(f"reducing embeddings dimension with PCA keeping {total_variance_explained * 100}% of variance explained...")
	pca = PCA(n_components=total_variance_explained)

	embeddings_reduced = pca.fit_transform(embeddings_df)
	print(f"successfully reduced from {embeddings_df.shape[1]} features to {reduced_embeddings_df.shape[1]} features")

	embeddings_reduced_df = pd.DataFrame(embeddings_reduced, index = embeddings_df.index)
	embeddings_reduced_df = embeddings_reduced_df.add_prefix(prefix)

	return embeddings_reduced_df