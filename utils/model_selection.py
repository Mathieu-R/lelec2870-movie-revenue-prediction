import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_predict, validation_curve, learning_curve

from skopt import BayesSearchCV

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer

def linreg(X_train, y_train, X_test, y_test):
	lr = LinearRegression()

	# train the model on training set
	lr.fit(X_train, y_train)

	# predict values with testing set
	y_pred = lr.predict(X_test)

	# compare predicted values with the testing target using mse
	rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
	return rmse

def perform_grid_search(model, hyperparameters, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=kf, scoring=scorer, n_jobs=-1)

	grid_search.fit(X_train, y_train)
	best_score = grid_search.score(X_test, y_test)

	return grid_search.best_estimator_, grid_search.best_params_, best_score

def perform_random_search(model, hyperparameters, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	random_search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=40, cv=kf, scoring=scorer, n_jobs=-1)

	random_search.fit(X_train, y_train)
	best_score = random_search.score(X_test, y_test)

	return random_search.best_estimator_, random_search.best_params_, best_score

def perform_bayesian_search(model, hyperparameters, n_iter, X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	bayesion_search = BayesSearchCV(estimator=model, search_spaces=hyperparameters, n_iter=n_iter, cv=kf, scoring=scorer, n_jobs=-1)

	bayesion_search.fit(X_train, y_train)
	best_score = bayesion_search.score(X_test, y_test)

	return bayesion_search.best_estimator_, bayesion_search.best_params_, best_score

# to know how many data you need 
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, kf, scorer):
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.2, 1.0, 5), cv=kf, scoring=scorer)

	train_mean = np.mean(train_scores, axis=1)
	train_std = np.std(train_scores, axis=1)

	val_mean = np.mean(val_scores, axis=1)
	val_std = np.std(val_scores, axis=1)

	plt.figure(figsize=(12, 8))

	plt.plot(train_sizes, train_mean, color="blue", marker="o", markersize=5, label="train score")
	plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")

	plt.plot(train_sizes, val_mean, color="green", linestyle="--", marker="s", markersize=5, label="validation score")
	plt.fill_between(train_sizes, val_mean + val_std, val_mean - val_std, alpha=0.15, color="green")

	plt.title(f"learning curve: {model_name}")
	plt.xlabel("train sizes")
	plt.ylabel("score (RMSE)")
	plt.legend()

# detect overfitting
def validate_model(model, model_name, param_name, param_range, X_train, y_train, X_test, y_test, kf, scorer):
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	train_scores, val_scores = validation_curve(model, X_train, y_train, param_name=param_name, param_range=param_range, cv=kf, scoring=scorer)

	train_mean = np.mean(train_scores, axis=1)
	train_std = np.std(train_scores, axis=1)

	val_mean = np.mean(val_scores, axis=1)
	val_std = np.std(val_scores, axis=1)

	plt.figure(figsize=(12, 8))

	plt.plot(param_range, train_mean, color="blue", marker="o", markersize=5, label="train score")
	plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")

	plt.plot(param_range, val_mean, color="green", linestyle="--", marker="s", markersize=5, label="validation score")
	plt.fill_between(param_range, val_mean + val_std, val_mean - val_std, alpha=0.15, color="green")

	plt.title(f"validation curve: {model_name}")
	plt.xlabel(param_name)
	plt.ylabel("score (RMSE)")
	plt.legend()

def compare_models(models, X_train, y_train, X_test, y_test, kf, scorer):
	print("COMPARING MODELS...")
	
	lr_score = linreg(X_train, y_train, X_test, y_test)
	
	print("Linear Regression RMSE: {:.3f}".format(lr_score))

	for model_name, model_params in models.items():
		best_estimator, best_params, best_score = perform_bayesian_search(
			model=model_params["instance"],
			hyperparameters=model_params["hyperparameters"],
			n_iter=model_params["n_iter"],
			X_train=X_train, 
			y_train=y_train, 
			X_test=X_test, 
			y_test=y_test,
			kf=kf,
			scorer=scorer
		)

		print(model_name)
		print(best_params)

		val_param_name = model_params["validation_param"]
		val_param_range = model_params["hyperparameters"][val_param_name]

		evaluate_model(best_estimator, model_name, X_train, y_train, X_test, y_test, kf, scorer)
		validate_model(best_estimator, model_name, val_param_name, val_param_range, X_train, y_train, X_test, y_test, kf, scorer)

		print("{} RMSE: {:.3f}".format(model_name, best_score))