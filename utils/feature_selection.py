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
from sklearn.feature_selection import RFE

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

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

def select_features_MI(X_train, y_train, X_test, percentile=80):
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

def select_features_RFE(X_train, y_train, X_test):
	print("-" * 25)
	print("FEATURE SELECTION (RANDOM FEATURE ELIMINATION)...")
	print("-" * 25)

	lcv = LassoCV()
	lcv.fit(X_train, y_train)

	lcv_mask = lcv.coef_ != 0
	lcv_k_features_keeped = sum(lcv_mask)
	#print(lcv.score(X_test, y_test))

	rf = RandomForestRegressor()
	rfe_rf = RFE(rf, n_features_to_select=lcv_k_features_keeped, step=5, verbose=False)
	rfe_rf.fit(X_train, y_train)

	rf_mask = rfe_rf.support_

	gb = GradientBoostingRegressor()
	rfe_gb = RFE(gb, n_features_to_select=lcv_k_features_keeped, step=5, verbose=False)
	rfe_gb.fit(X_train, y_train)

	gb_mask = rfe_gb.support_

	votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)
	votes_mask = votes >= 2

	X_train_filtered = X_train.loc[:, votes_mask]
	X_test_filtered = X_test.loc[:, votes_mask]

	print(f"reduced from {X_train.shape[1]} features to {X_train_filtered.shape[1]} features")

	return X_train_filtered, X_test_filtered