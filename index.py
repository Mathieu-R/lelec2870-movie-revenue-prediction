"""
After doing multiple analysis in a Jupyter Notebook. 
I switch to plain Python as this project is too much computational intensive for a Notebook.
"""

import pandas as pd 
import seaborn as sns 
import numpy as np 

import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel

from sklearn.model_selection import train_test_split, KFold, cross_val_predict

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error

import ast

class MovieRevenuePredictor():
	def __init__(self) -> None:
		"""Load the datasets
			X1: entry dataset
			Y1: target dataset

			X2: entry dataset used to make prediction
		"""
		self.X1 = pd.read_csv("datasets/X1.csv")
		self.Y1 = pd.read_csv("datasets/X2.csv", header=None, names=["revenues"])

		self.X2 = pd.read_csv("datasets/X2.csv")

		# drop unecessary column `Unnamed: 0`
		self.X1.drop("Unnamed: 0", axis=1, inplace=True)

		# for feature engineering and the sake of simplicity, we concatenate X1 with its target Y1
		self.df = pd.concat([self.X1, self.Y1], axis=1)

		print(f"X1 dataset contains {self.X1.shape[0]} observations and {self.X1.shape[1]} features")
		print(f"X2 dataset (for prediction only) contains {self.X2.shape[0]} observations")

		print(f"features: {list(self.X1.columns)}")
		print(f"target: {list(self.Y1.columns)}")

		df = self.preprocess(self.df)
		X2 = self.preprocess(self.X2)

		X = df.drop("revenues", axis=1)
		Y = df["revenues"]

		self.special_preprocess(X, Y)

	def preprocess(self, df):
		print("PREPROCESSING 1/2...")
		print("--------------------------")
		self.preprocess_duplicated_and_missing(df)
		self.preprocess_irrelevant_features(df)

		df = self.one_hot_encode_genres_feature(df)
		df = self.label_encode_studio_feature(df)
		df = self.other_fixes(df)
		return df

	def special_preprocess(self, X, Y):
		print("PREPROCESSING 2/2...")
		print("--------------------------")

		X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, shuffle = True, random_state = 0)

		print(f"training dataset dimension: X_train: {X_train.shape}, y_train: {y_train.shape}")
		print(f"testing dataset dimension: X_test: {X_test.shape}, y_test: {y_test.shape}")

		# remove outliers only on train set
		# as test set should be representative of the reality
		self.remove_outliers(X_train)

		# extract feature vectors
		X_train_img_embeddings = self.extract_embeddings_features(X_train["img_embeddings"])
		X_test_img_embeddings = self.extract_embeddings_features(X_test["img_embeddings"])

		X_train_text_embeddings = self.extract_embeddings_features(X_train["text_embeddings"])
		X_test_text_embeddings = self.extract_embeddings_features(X_test["text_embeddings"])

		X_train_img_df, X_test_img_df = self.pca_on_embeddings(X_train_img_embeddings, X_test_img_embeddings, X_train.index, X_test.index, prefix="img_feature", total_variance_explained=0.8)

		X_train_text_df, X_test_text_df = self.pca_on_embeddings(X_train_text_embeddings, X_test_text_embeddings, X_train.index, X_test.index, prefix="text_feature", total_variance_explained=0.8)

		X_train = pd.concat([X_train, X_train_img_df, X_train_text_df], axis=1)
		X_test = pd.concat([X_test, X_test_img_df, X_test_text_df])

		# should also extract features for X2

		# standardize
		X_train, X_test = self.standardize(X_train, X_test)


	def preprocess_duplicated_and_missing(self, df):
		# duplicated values
		print("# duplicated features:")
		print({df[df.duplicated(subset=df.columns.difference(["revenues"]))].count().to_string()})

		df.drop_duplicates(subset=df.columns.difference(["revenues"]), keep="first", inplace=True)

		# missing values
		# this dataset use the special character "\N" for missing values. 

		# replace "\N" by "-1" (only for conversion to int)
		df.replace("\\N", "-1", inplace=True)

		print(df["runtime"].unique())

		# convert runtime feature to int type
		df["runtime"] = df["runtime"].astype(int)

		# replace -1 (for column runtime) and "-1" (for column genres) by NaN
		df.replace(-1, np.nan, inplace=True)
		df.replace("-1", np.nan, inplace=True)

		print("percentage of missing values:")
		print((df.isna().sum() / df.shape[0] * 100).round(decimals=2))

		# drop observations with missing genres
		df.dropna(subset=["genres"], axis=0, inplace=True)

		# impute observations with missing runtime (because > 5% of missing values)
		# replace by the mean (since runtime feature is not too for from a Gaussian)
		df["runtime"].fillna(df["runtime"].mean(), inplace=True)

		print("[X] duplicated and missing values")

	def preprocess_irrelevant_features(self, df):
		# drop `img_url` and `description` since we have the embeddings
		df.drop(["img_url", "description"], axis=1, inplace=True)

		# we do not have movies for mature audience so feature `is_adult` has variance 0
		df.drop(["is_adult"], axis=1, inplace=True)

		print("[X] irrelevant features")

	def one_hot_encode_genres_feature(self, df):
		# separate all genres into one big list of list of genres
		genres_list = df["genres"].str.split(",").tolist()

		unique_genres = []

		# retrieve each genre
		for sublist in genres_list:
			for genre in sublist:
				if genre not in unique_genres:
					unique_genres.append(genre)

		# sort
		unique_genres = sorted(unique_genres)

		# one hot encode movies genres
		df = df.reindex(df.columns.tolist() + unique_genres, axis = 1, fill_value = 0)

		for index, row in df.iterrows():
			for genre in row["genres"].split(","):
				df.loc[index, genre] = 1

		# drop old genres column
		df.drop("genres", axis=1, inplace=True)

		print("[X] One-Hot encoding")
		return df

	def label_encode_studio_feature(self, df):
		self.label_encoder_studio = LabelEncoder()

		df["studio"] = self.label_encoder_studio.fit_transform(df["studio"].to_numpy())

		print("[X] Label encoding")
		return df

	def other_fixes(self, df):
		# it does not make sens to have `release_year` and `n_votes` features as type float
		# let's convert them into int
		df["release_year"] = df["release_year"].astype(int)
		df["n_votes"] = df["n_votes"].astype(int)

		# in notebook, I dropped this feature at the end of first part of preprocessing
		# but I'm not sure why
		df.drop("title", axis=1, inplace=True)

		print("[X] minor fixes")
		return df


	def remove_outliers(self, X_train):
		pass

	def standardize(self, X_train, X_test):
		scaler = RobustScaler()

		# fit the scaler on training dataset
		X_train_scaled = scaler.fit_transform(X_train)

		# apply the scaler on testing dataset (and so avoid introducing bias)
		X_test_scaled = scaler.transform(X_test)

		# should do the same on X2 

		X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
		X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

		return X_train, X_test
	
	def extract_embeddings_features(self, embeddings):
		"""
		Args:
			embeddings (pd.Series): Pandas Series of string representation of feature vectors.
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

	def pca_on_embeddings(self, train_embeddings_matrix, test_embeddings_matrix, train_index, test_index, prefix, total_variance_explained = 0.95):
		scaler = StandardScaler()

		n_features_before_pca = len(train_embeddings_matrix[0])

		# standardize data
		train_embeddings_matrix = scaler.fit_transform(train_embeddings_matrix)
		test_embeddings_matrix = scaler.transform(test_embeddings_matrix)

		pca = PCA(n_components=total_variance_explained)

		# run pca
		train_embeddings_matrix = pca.fit_transform(train_embeddings_matrix)
		test_embeddings_matrix = pca.transform(test_embeddings_matrix)

		print(f"successfully reduced from {n_features_before_pca} features to {len(train_embeddings_matrix[0])} features keeping {total_variance_explained * 100}% of variance explained")

		train_embeddings_df = pd.DataFrame(train_embeddings_matrix, index=train_index).add_prefix(prefix)
		test_embeddings_df = pd.DataFrame(test_embeddings_matrix, index=test_index).add_prefix(prefix)

		return scaler.inverse_transform(train_embeddings_df), scaler.inverse_transform(test_embeddings_df)


if __name__ == "__main__":
	predictor = MovieRevenuePredictor()