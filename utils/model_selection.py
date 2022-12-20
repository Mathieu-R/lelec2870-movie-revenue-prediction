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

def perform_grid_search(model, hyperparameters, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=kf, scoring=scorer, n_jobs=-1, error_score='raise')

	grid_search.fit(X_train, y_train)
	best_score = grid_search.score(X_test, y_test)

	return grid_search.best_estimator_, grid_search.best_params_, -best_score

def perform_random_search(model, hyperparameters, n_iter, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	random_search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=n_iter, cv=kf, scoring=scorer, n_jobs=-1, error_score="raise")

	random_search.fit(X_train, y_train)
	best_score = random_search.score(X_test, y_test)

	return random_search.best_estimator_, random_search.best_params_, -best_score

class ModelSelection():
	def __init__(self, X_train, y_train, X_test, y_test, kf, scorer) -> None:
		self.X_train = X_train 
		self.y_train = y_train
		self.X_test = X_test 
		self.y_test = y_test 
		self.kf = kf
		self.scorer = scorer

	def perform_bayesian_search(self, model, hyperparameters, n_iter):
		self.bayesian_search = BayesSearchCV(
			estimator=model, 
			search_spaces=hyperparameters, 
			cv=self.kf, 
			scoring=self.scorer, 
			# allows to compute a score on the test data
			refit=True,
			# include training scores in cv_results_
			return_train_score=True, 
			n_iter=n_iter, 
			n_jobs=-1,
			random_state=42
		)

		# def status_print(optim_results):
		# 	# get all models tested so far
		# 	all_models = pd.DataFrame(self.bayesian_search.cv_results_)

		# 	# get current parameters and the best parameters
		# 	best_params = pd.Series(self.bayesian_search.best_params_)

		# 	print('Model #{}\nBest score: {}\nBest params: {}\n'.format(
		# 		len(all_models),
		# 		np.round(self.bayesian_search.best_score_, 3),
		# 		self.bayesian_search.best_params_
		# 	))


		self.bayesian_search.fit(self.X_train, self.y_train)
		test_score = -self.bayesian_search.score(self.X_test, self.y_test)

		return self.bayesian_search

		return self.bayesian_search.best_estimator_, self.bayesian_search.best_params_, -test_score

	
	def test_model(self, model, name):
		searcher = self.perform_bayesian_search(
			model=model["instance"], 
			hyperparameters=model["hyperparameters"], 
			n_iter=model["n_iter"]
		)

		return searcher