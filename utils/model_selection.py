import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict, validation_curve, learning_curve
from skopt import BayesSearchCV

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

def linreg(X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	lr = TransformedTargetRegressor(LinearRegression(), func=np.log1p, inverse_func=np.expm1)
	val_scores = cross_val_score(estimator=lr, X=X_train, y=y_train, scoring=scorer, cv=kf, n_jobs=-1)
	mean_val_score = -val_scores.mean()

	# train the model on training set
	lr.fit(X_train, y_train)

	# predict values with testing set
	y_pred = lr.predict(X_test)

	# compare predicted values with the testing target using mse
	rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
	r2 = r2_score(y_true=y_test, y_pred=y_pred)

	#return rmse
	return mean_val_score, rmse, r2

class ModelSelection():
	def __init__(self, X_train, y_train, X_test, y_test, kf, scorer) -> None:
		self.X_train = X_train 
		self.y_train = y_train
		self.X_test = X_test 
		self.y_test = y_test 
		self.kf = kf
		self.scorer = scorer

	def perform_search(self, model, search_type):
		if search_type == "gs":
			return self.perform_grid_search(model)
		elif search_type == "rs":
			return self.random_search(model)
		elif search_type == "bs":
			return self.perform_bayesian_search(model)
		else:
			print("Unknown search type...")

	def perform_bayesian_search(self, model):
		self.bayesian_search = BayesSearchCV(
			estimator=model["instance"], 
			search_spaces=model["hyperparameters"], 
			cv=self.kf, 
			scoring=self.scorer, 
			# allows to compute a score on the test data
			refit=True,
			# include training scores in cv_results_
			return_train_score=True, 
			n_iter=model["n_iter"], 
			n_jobs=-1,
			random_state=42
		)

		self.bayesian_search.fit(self.X_train, self.y_train)
		return self.bayesian_search

	def perform_random_search(self, model):
		self.random_search = RandomizedSearchCV(
			estimator=model["instance"], 
			param_distributions=model["hyperparameters"], 
			cv=self.kf, 
			scoring=self.scorer, 
			# allows to compute a score on the test data
			refit=True,
			# include training scores in cv_results_
			return_train_score=True,
			n_iter=model["n_iter"], 
			n_jobs=-1,
			random_state=42
		)

		self.random_search(self.X_train, self.y_train)
		return self.random_search
	
	def perform_grid_search(self, model):
		self.grid_search = GridSearchCV(
			estimator=model["instance"], 
			param_grid=model["hyperparameters"], 
			cv=self.kf, 
			scoring=self.scorer, 
			# allows to compute a score on the test data
			refit=True,
			# include training scores in cv_results_
			return_train_score=True,
			error_score=0,
			n_jobs=-1
		)

		self.grid_search.fit(self.X_train, self.y_train)
		return self.grid_search