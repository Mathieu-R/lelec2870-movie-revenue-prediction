""" 
We want to remove redundant or irrelevant features to improve computational efficiency and reduce the risk of overfitting.
As I understand, there's multiple ways to selection features via filter method. As a reminder, filter method is independent of any machine learning model but does not take into account feature redundancy. 

Some of them are:
- `Chi-Square` and `ANOVA`: for categorical variables and categorical targets    
- `Correlation matrix`: for continuous variables, continuous target and linear model    
- `Mutual information`: for continuous variables, continuous target and non-linear model     

Since we're dealing with continuous target and we will train linear and non-linear models, we use the two last one.
"""

import pandas as pd 
import numpy as np 

from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestRegressor

def get_discrete_features_mask(X_train):
	CONTINUOUS_FEATURES = ["ratings", "n_votes", "production_year", "runtime", "release_year","studio"]
	discrete_features_mask = []

	for column in X_train.columns:
		discrete_features_mask.append(column not in CONTINUOUS_FEATURES)

	return discrete_features_mask

def select_features_correlation(X_train, y_train, X_test, percentile=80):
	print("-" * 25)
	print("FEATURE SELECTION (CORRELATION MATRIX)...")
	print("-" * 25)

	# select k best features according to correlation matrix
	percentile_best = SelectPercentile(score_func=f_regression, percentile=percentile)
	X_train_filtered = percentile_best.fit_transform(X_train, y_train)
	X_test_filtered = percentile_best.transform(X_test)

	best_features = percentile_best.get_feature_names_out()

	X_train_filtered = pd.DataFrame(X_train_filtered, columns=best_features, index=X_train.index)
	X_test_filtered = pd.DataFrame(X_test_filtered, columns=best_features, index=X_test.index)

	print(f"reduced from {X_train.shape[1]} features to {X_train_filtered.shape[1]} features")
	
	return X_train_filtered, X_test_filtered

# https://towardsdatascience.com/feature-selection-techniques-for-classification-and-python-tips-for-their-application-10c0ddd7918b
def mrmr(X_train, y_train):
	X_train_copy = X_train.copy()
	y_train_copy = y_train.copy()

	# relevancy of input features with the continuous target
	relevancies = mutual_info_regression(X_train_copy, y_train_copy)

	redundancies = []
	for index, data in X_train_copy.items():
		# redundancy of input feature "i" with all other input features
		target = X_train_copy.loc[:, index]
		input = X_train_copy.drop(columns=index)
		redundancy = mutual_info_regression(input, target)
		redundancies.append(redundancy.sum() / input.shape[1])
	
	# compute score
	scores = relevancies - np.abs(redundancies)
	
	idx_sorted = np.argsort(scores)[::-1]
	sorted_scores = scores[idx_sorted]
	sorted_columns = X_train.columns[idx_sorted].values

	return sorted_scores, sorted_columns

def get_mutual_information_matrix(X_train):
	p = len(X_train.columns)
	
	MI_matrix = np.zeros((p,p))

	for i in range(p):
		for j in range(p):
			# triangular matrix
			if i < j:
				continue
			# put 1 in diagonal to avoid dividing by 0 when normalizing
			elif i == j:
				MI_matrix[i,j] = 1
			# print(i,j)
			# # if 2 features are discretes
			# if discrete_features[i] and discrete_features[j]:
			# 	MI_matrix[i,j] = mutual_info_score(X_train.iloc[:,i], X_train.iloc[:,j])
			# # if 1 feature is discerte
			# elif not discrete_features[i] and discrete_features[j]:
			# 	MI_matrix[i,j] = mutual_info_classif(X_train.iloc[:,i].to_frame(), X_train.iloc[:,j], discrete_features=[False])[0]
			# elif discrete_features[i] and not discrete_features[j]:
			# 	MI_matrix[i,j] = mutual_info_classif(X_train.iloc[:,j].to_frame(), X_train.iloc[:,i], discrete_features=[False])[0]
			# # if 0 feature are discretes
			# else:
			else:
				MI_matrix[i,j] = mutual_info_regression(X_train.iloc[:,i].to_frame(), X_train.iloc[:,j], discrete_features=[False])[0]
	return MI_matrix

def normalize_mutual_information_matrix(MI_matrix):
	p = len(MI_matrix[0])

	# normalize between 0 and 1
	diag = np.copy(np.diag(MI_matrix))
	for i in range(p):
		for j in range(p):
			if i < j:
				continue 
			MI_matrix[i,j] = MI_matrix[i,j] / np.sqrt(diag[i] * diag[j])
	
	return MI_matrix

def select_features_MI_percentile(X_train, y_train, X_test, percentile=80):
	print("-" * 25)
	print("FEATURE SELECTION (MUTUAL INFORMATION)...")
	print("-" * 25)
	
	# select k best features according to MI
	percentile_best = SelectPercentile(score_func=mutual_info_regression, percentile=percentile)
	X_train_filtered = percentile_best.fit_transform(X_train, y_train)
	X_test_filtered = percentile_best.transform(X_test)

	best_features = percentile_best.get_feature_names_out()

	X_train_filtered = pd.DataFrame(X_train_filtered, columns=best_features, index=X_train.index)
	X_test_filtered = pd.DataFrame(X_test_filtered, columns=best_features, index=X_test.index)

	print(f"reduced from {X_train.shape[1]} features to {X_train_filtered.shape[1]} features")
	
	return X_train_filtered, X_test_filtered

def select_features_MI_kbest(X_train, y_train, X_test, k=20):
	print("-" * 25)
	print("FEATURE SELECTION (MUTUAL INFORMATION)...")
	print("-" * 25)
	
	# select k best features according to MI
	k_best = SelectKBest(score_func=mutual_info_regression, k=k)
	X_train_filtered = k_best.fit_transform(X_train, y_train)
	X_test_filtered = k_best.transform(X_test)

	best_features = k_best.get_feature_names_out()

	X_train_filtered = pd.DataFrame(X_train_filtered, columns=best_features, index=X_train.index)
	X_test_filtered = pd.DataFrame(X_test_filtered, columns=best_features, index=X_test.index)

	print(f"reduced from {X_train.shape[1]} features to {X_train_filtered.shape[1]} features")
	
	return X_train_filtered, X_test_filtered

def select_features_RFECV(X_train, y_train, X_test, kf, scorer):
	print("-" * 25)
	print("FEATURE SELECTION (RANDOM FEATURE ELIMINATION)...")
	print("-" * 25)

	all_features = X_train.columns.tolist()
	print(all_features)

	rf = RandomForestRegressor(random_state=42)

	rfe = RFECV(rf, cv=kf, scoring=scorer)
	rfe.fit(X_train, y_train)

	selected_features = np.array(all_features)[rfe.get_support()]
	print(selected_features)

	X_train_filtered = X_train[selected_features]
	X_test_filtered = X_test[selected_features]

	return X_train_filtered, X_test_filtered