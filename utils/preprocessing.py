import pandas as pd 
import numpy as np 

import category_encoders as ce

from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_duplicated_and_missing(df):
	# drop duplicated values
	df.drop_duplicates(subset=df.columns.difference(["revenues"]), keep="first", inplace=True)

	# drop observations with missing genres
	df.dropna(subset=["genres"], axis=0, inplace=True)

	# impute observations with missing runtime (because > 5% of missing values)
	# replace by the mean (since runtime feature is not too for from a Gaussian)
	df["runtime"].fillna(df["runtime"].mean(), inplace=True)

	print("[X] Removing duplicated and missing values")
	return df

def preprocess_irrelevant_features(df):
	# drop `img_url` and `description` since we have the embeddings
	df.drop(["img_url", "description"], axis=1, inplace=True)

	# we do not have movies for mature audience so feature `is_adult` has variance 0
	df.drop(["is_adult"], axis=1, inplace=True)

	print("[X] Removing irrelevant features")
	return df

def one_hot_encode_genres_feature(df):
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
	df = df.reindex(df.columns.tolist() + unique_genres, axis=1, fill_value=0)

	for index, row in df.iterrows():
		for genre in row["genres"].split(","):
			df.loc[index, genre] = 1

	# drop old genres column
	df.drop("genres", axis=1, inplace=True)

	print("[X] One-Hot encoding")
	return df

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

	if "revenues" in df:
		df["revenues"] = np.log(df["revenues"])

	# in notebook, I dropped this feature at the end of first part of preprocessing
	# but I'm not sure why
	df.drop("title", axis=1, inplace=True)

	print("[X] Minor fixes")
	return df

def remove_outliers(X_train):
	pass

def standardize(X_train, X_test):
	standard_scaler = StandardScaler()

	# fit the scaler on training dataset
	X_train_scaled = standard_scaler.fit_transform(X_train)

	# apply the scaler on testing dataset (and so avoid introducing bias)
	X_test_scaled = standard_scaler.transform(X_test)

	# should do the same on X2 

	X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
	X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

	return X_train, X_test, standard_scaler