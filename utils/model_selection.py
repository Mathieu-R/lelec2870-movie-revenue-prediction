import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_halving_search_cv

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict, validation_curve, learning_curve, HalvingRandomSearchCV

from skopt import BayesSearchCV

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer

from utils.plots import evaluate_model, validate_model

def linreg(X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	lr = LinearRegression()
	scores = cross_val_score(estimator=lr, X=X_train, y=y_train, cv=kf, scoring=scorer, n_jobs=-1)

	# train the model on training set
	#lr.fit(X_train, y_train)

	# predict values with testing set
	#y_pred = lr.predict(X_test)

	# compare predicted values with the testing target using mse
	#rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
	#return rmse
	return scores.mean()

def perform_grid_search(model, hyperparameters, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=kf, scoring=scorer, n_jobs=-1, error_score='raise')

	grid_search.fit(X_train, y_train)
	best_score = grid_search.score(X_test, y_test)

	return grid_search.best_estimator_, grid_search.best_params_, best_score

def perform_random_search(model, hyperparameters, n_iter, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	random_search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=n_iter, cv=kf, scoring=scorer, n_jobs=-1, error_score="raise")

	random_search.fit(X_train, y_train)
	best_score = random_search.score(X_test, y_test)

	return random_search.best_estimator_, random_search.best_params_, best_score

def perform_halving_random_search(model, hyperparameters, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	random_search = HalvingRandomSearchCV(estimator=model, param_distributions=hyperparameters, n_candidates="exhaust", resource="n_samples", factor=2, cv=kf, scoring=scorer, refit=True, error_score="raise", random_state=0, n_jobs=-1)

	random_search.fit(X_train, y_train)
	best_score = random_search.score(X_test, y_test)

	return random_search.best_estimator_, random_search.best_params_, best_score

def perform_bayesian_search(model, hyperparameters, n_iter, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	bayesion_search = BayesSearchCV(estimator=model, search_spaces=hyperparameters, n_iter=n_iter, cv=kf, scoring=scorer, n_jobs=-1)

	bayesion_search.fit(X_train, y_train)
	best_score = bayesion_search.score(X_test, y_test)

	return bayesion_search.best_estimator_, bayesion_search.best_params_, best_score

def compare_models(models, X_train, y_train, X_test, y_test, kf, scorer):
	print("-" * 25)
	print("COMPARING MODELS...")
	print("-" * 25)

	print("+" * 25)
	print("Linear Regression")
	print("+" * 25)
	
	lr_score = linreg(X_train, y_train, X_test, y_test, kf, scorer)
	
	print("Linear Regression RMSE: {:.3f}".format(lr_score))

	for model_name, model_params in models.items():
		best_estimator, best_params, best_score = perform_halving_random_search(
			model=model_params["instance"],
			hyperparameters=model_params["hyperparameters"],
			X_train=X_train, 
			y_train=y_train, 
			X_test=X_test, 
			y_test=y_test,
			kf=kf,
			scorer=scorer
		)

		print("+" * 25)
		print(model_name)
		print("+" * 25)

		print(best_params)

		val_param_name = model_params["validation_param"]
		val_param_range = model_params["hyperparameters"][val_param_name]

		evaluate_model(best_estimator, model_name, X_train, y_train, X_test, y_test, kf, scorer)
		validate_model(best_estimator, model_name, val_param_name, val_param_range, X_train, y_train, X_test, y_test, kf, scorer)

		print("{} RMSE: {:.3f}".format(model_name, best_score))

def test_model(model, name, X_train, y_train, X_test, y_test, kf, scorer):
	print(model["instance"])
	best_estimator, best_params, best_score = perform_random_search(
		model=model["instance"], 
		hyperparameters=model["hyperparameters"], 
		n_iter=model["n_iter"],
		X_train=X_train, 
		y_train=y_train, 
		X_test=X_test, 
		y_test=y_test, 
		kf=kf
	)

	print(best_params)

	val_param_name = model["validation_param"]
	val_param_range = model["hyperparameters"][val_param_name]

	#evaluate_model(best_estimator, name, X_train, y_train, X_test, y_test, kf, scorer)
	#validate_model(best_estimator, name, val_param_name, val_param_range, X_train, y_train, X_test, y_test, kf, scorer)

	print("{} RMSE: {:.3f}".format(name, best_score))

	return best_estimator, best_params, best_score