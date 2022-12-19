import pandas as pd 
import numpy as np 

import category_encoders as ce

from scipy.stats import zscore

from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_duplicated_and_missing(df, train):
	# drop duplicated values
	df.drop_duplicates(subset=df.columns.difference(["revenues"]), keep="first", inplace=True)

	# drop observations with missing genres
	df.dropna(subset=["genres"], axis=0, inplace=True)

	# impute observations with missing runtime (because > 5% of missing values)
	# replace by the mean (since runtime feature is not too for from a Gaussian)
	# use only the data from the "train set" (X1.csv) in order to avoid data leakage
	df["runtime"].fillna(train["runtime"].mean(), inplace=True)

	print("[X] Removing duplicated and missing values")
	return df

def preprocess_irrelevant_features(df):
	# drop `img_url` and `description` since we have the embeddings
	df.drop(["img_url", "description"], axis=1, inplace=True)

	# we do not have movies for mature audience so feature `is_adult` has variance 0
	df.drop(["is_adult"], axis=1, inplace=True)

	print("[X] Removing irrelevant features")
	return df

def one_hot_encode_genres_feature(df, X2):
	def preprocess_genres(genre_list):
		return str(genre_list).split(",")

	df["genres_preprocessed"] = df["genres"].apply(lambda x: preprocess_genres(x))
	X2["genres_preprocessed"] = X2["genres"].apply(lambda x: preprocess_genres(x))

	genres_dict = dict()

	for genre_list in df["genres_preprocessed"]:
		for genre in genre_list:
			if genre not in genres_dict:
				genres_dict[genre] = 1
			else:
				genres_dict[genre] += 1

	genres_df = pd.DataFrame.from_dict(genres_dict, columns=["number_of_movies"], orient="index")
	genres_df = genres_df.sort_values(by="number_of_movies", ascending=False)

	for genre in genres_df.index.values:
		df["genre_" + genre] = df["genres_preprocessed"].apply(lambda x: 1 if genre in x else 0)
		X2["genre_" + genre] = X2["genres_preprocessed"].apply(lambda x: 1 if genre in x else 0)

	# drop old columns
	df.drop("genres", axis=1, inplace=True)
	X2.drop("genres", axis=1, inplace=True)
	df.drop("genres_preprocessed", axis=1, inplace=True)
	X2.drop("genres_preprocessed", axis=1, inplace=True)

	print("[X] One-Hot encoding genres feature")
	return df, X2

def one_hot_encode_studio_feature(df, X2):
	# frequency of each studio
	studio_freq = df["studio"].value_counts(normalize=True, ascending=True)

	# replace each studio by its frequency
	mapping = df["studio"].map(studio_freq)

	# replace studio representing less than 1% of all studios by "other"
	# keep a list of all kept studios
	kept_studios = df["studio"].mask(mapping < 0.01, "other").unique()

	for studio in kept_studios:
		df["studio_" + studio] = df["studio"].apply(lambda x: 1 if studio in x else 0)
		X2["studio_" + studio] = df["studio"].apply(lambda x: 1 if studio in x else 0)

	# drop old columns
	df.drop("studio", axis=1, inplace=True)
	X2.drop("studio", axis=1, inplace=True)

	print("[X] One-Hot encoding studio feature")
	return df, X2

def label_encode_studio_feature(df):
	label_encoder_studio = LabelEncoder()

	df["studio"] = label_encoder_studio.fit_transform(df["studio"].to_numpy())

	print("[X] Label encoding")
	return df

def count_encode_studio_feature(df):
	count_encoder_studio = ce.CountEncoder()

	df["studio"] = count_encoder_studio.fit_transform(df["studio"])

	print("[X] Count encoding")
	return df

def other_fixes(df):
	# it does not make sense to have `release_year` and `n_votes` features as type float
	# let's convert them into int
	df["release_year"] = df["release_year"].astype(int)
	#df["n_votes"] = df["n_votes"].astype(int)

	# log-transform `n_votes` and `revenues` features to fix the skweness
	df["n_votes"] = np.log(df["n_votes"])

	# target variable is automatically transformed / untransformed
	# with TransformedTargetRegressor
	#if "revenues" in df:
	#	df["revenues"] = np.log(df["revenues"])

	# in notebook, I dropped this feature at the end of first part of preprocessing
	# but I'm not sure why
	df.drop("title", axis=1, inplace=True)

	print("[X] Minor fixes")
	return df

def remove_outliers(X_train, y_train, columns):
	z_score = np.abs(zscore(X_train[columns]))
	filtered_entries = (z_score < 3).all(axis=1)

	X_train = X_train[filtered_entries]
	y_train = y_train[filtered_entries]

	return X_train, y_train

def standardize(X_train, X_test, X2):
	standard_scaler = StandardScaler()

	# fit the scaler on training dataset
	X_train_scaled = standard_scaler.fit_transform(X_train)

	# apply the scaler on testing dataset (and so avoid introducing bias)
	X_test_scaled = standard_scaler.transform(X_test)
	X2_scaled = standard_scaler.transform(X2)

	# should do the same on X2 

	X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
	X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
	X2 = pd.DataFrame(X2_scaled, columns=X2.columns, index=X2.index)

	return X_train, X_test, X2, standard_scaler