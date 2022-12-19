import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_halving_search_cv

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict, validation_curve, learning_curve, HalvingRandomSearchCV

from skopt import BayesSearchCV

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from utils.plots import evaluate_model, validate_model

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

def perform_halving_random_search(model, hyperparameters, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	random_search = HalvingRandomSearchCV(estimator=model, param_distributions=hyperparameters, n_candidates="exhaust", resource="n_samples", factor=2, cv=kf, scoring=scorer, refit=True, error_score="raise", random_state=0, n_jobs=-1)

	random_search.fit(X_train, y_train)
	best_score = random_search.score(X_test, y_test)

	return random_search.best_estimator_, random_search.best_params_, -best_score

def perform_bayesian_search(model, hyperparameters, n_iter, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	bayesian_search = BayesSearchCV(
		estimator=model, 
		search_spaces=hyperparameters, 
		cv=kf, 
		scoring=scorer, 
		refit=True,
		n_iter=n_iter, 
		n_jobs=-1,
		random_state=42
	)

	def status_print(optim_results):
		# get all models tested so far
		all_models = pd.DataFrame(bayesian_search.cv_results_)

		# get current parameters and the best parameters
		best_params = pd.Series(bayesian_search.best_params_)

		print('Model #{}\nBest score: {}\nBest params: {}\n'.format(
			len(all_models),
			np.round(bayesian_search.best_score_, 3),
			bayesian_search.best_params_
		))

	bayesian_search.fit(X_train, y_train, callback=status_print)
	best_score = bayesian_search.score(X_test, y_test)

	return bayesian_search.best_estimator_, bayesian_search.best_params_, -best_score

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
			refit=True,
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
		best_score = self.bayesian_search.score(self.X_test, self.y_test)

		print(best_score)

		return self.bayesian_search.best_estimator_, self.bayesian_search.best_params_, -best_score

	
	def test_model(self, model, name):
		best_estimator, best_params, best_score = perform_bayesian_search(
			model=model["instance"], 
			hyperparameters=model["hyperparameters"], 
			n_iter=model["n_iter"],
			X_train=self.X_train, 
			y_train=self.y_train, 
			X_test=self.X_test, 
			y_test=self.y_test, 
			kf=self.kf,
			scorer=self.scorer
		)

		val_param_name = model["validation_param"]
		val_param_range = model["hyperparameters"][val_param_name]

		print("{} RMSE: {:.3f}".format(name, best_score))

		return best_estimator, best_params, best_score

def test_model(model, name, X_train, y_train, X_test, y_test, kf, scorer):
	print(model["instance"])
	best_estimator, best_params, best_score = perform_bayesian_search(
		model=model["instance"], 
		hyperparameters=model["hyperparameters"], 
		n_iter=model["n_iter"],
		X_train=X_train, 
		y_train=y_train, 
		X_test=X_test, 
		y_test=y_test, 
		kf=kf,
		scorer=scorer
	)

	print(best_params)

	val_param_name = model["validation_param"]
	val_param_range = model["hyperparameters"][val_param_name]

	#evaluate_model(best_estimator, name, X_train, y_train, X_test, y_test, kf, scorer)
	#validate_model(best_estimator, name, val_param_name, val_param_range, X_train, y_train, X_test, y_test, kf, scorer)

	print("{} RMSE: {:.3f}".format(name, best_score))

	return best_estimator, best_params, best_score

# def compare_models(models, X_train, y_train, X_test, y_test, kf, scorer):
# 	print("-" * 25)
# 	print("COMPARING MODELS...")
# 	print("-" * 25)

# 	print("+" * 25)
# 	print("Linear Regression")
# 	print("+" * 25)
	
# 	lr_score = linreg(X_train, y_train, X_test, y_test, kf, scorer)
	
# 	print("Linear Regression RMSE: {:.3f}".format(lr_score))

# 	for model_name, model_params in models.items():
# 		best_estimator, best_params, best_score = perform_halving_random_search(
# 			model=model_params["instance"],
# 			hyperparameters=model_params["hyperparameters"],
# 			X_train=X_train, 
# 			y_train=y_train, 
# 			X_test=X_test, 
# 			y_test=y_test,
# 			kf=kf,
# 			scorer=scorer
# 		)

# 		print("+" * 25)
# 		print(model_name)
# 		print("+" * 25)

# 		print(best_params)

# 		val_param_name = model_params["validation_param"]
# 		val_param_range = model_params["hyperparameters"][val_param_name]

# 		evaluate_model(best_estimator, model_name, X_train, y_train, X_test, y_test, kf, scorer)
# 		validate_model(best_estimator, model_name, val_param_name, val_param_range, X_train, y_train, X_test, y_test, kf, scorer)

# 		print("{} RMSE: {:.3f}".format(model_name, best_score))