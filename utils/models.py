def cv_knn(X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	hyperparameters = {
		"n_neighbors": list(range(1,30)),
		"p": [1, 2],
		"weights": ["uniform", "distance"]
	}

	knn = KNeighborsRegressor()
	random_search_knn = RandomizedSearchCV(estimator=knn, param_distributions=hyperparameters, n_iter=40, scoring=scorer, cv=kf, n_jobs=-1)

	random_search_knn.fit(X_train, y_train)
	best_score = random_search_knn.score(X_test, y_test)

	return random_search_knn.best_estimator_, best_score

def cv_mlp(X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	hyperparameters = {
		"hidden_layer_sizes": [(100,75,50), (75,50,25),(50,25,10)],
		"activation": ["identity", "logistic", "tanh", "relu"],
		"alpha": 10.0 ** -np.arange(1, 7), # https://scikit-learn.org/stable/modules/neural_networks_supervised.html,
		"max_iter": np.linspace(10, 100, 10)
	}

	mlp = MLPRegressor()
	random_search_mlp = RandomizedSearchCV(estimator=mlp, param_distributions=hyperparameters, n_iter=80, cv=kf, scoring=scorer, n_jobs=-1)

	random_search_mlp.fit(X_train, y_train)
	best_score = random_search_mlp.score(X_test, y_test)

	return random_search_mlp.best_estimator_, best_score

def cv_rforest(X_train, y_train, X_test, y_test, kf, scorer = "neg_mean_squared_error"):
	hyperparameters = {
		"n_estimators": np.linspace(100, 1000, 10),
		"criterion": ["squared_error", "absolute_error", "poisson"],
		"max_depth": [3, 10, None] # none means unbounded max depth
	}

	rforest = RandomForestRegressor(criterion="gini", min_samples_split=30)
	random_search_rforest = RandomizedSearchCV(estimator=rforest, param_distributions=hyperparameters, n_iter=80, cv=kf, scoring=scorer, n_jobs=-1)

	random_search_rforest.fit(X_train, y_train)
	best_score = random_search_rforest.score(X_test, y_test)

	return random_search_rforest.best_estimator_, best_score