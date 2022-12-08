""" 
A feature perfectly correlated with another one does not apport any new informations but add complexity to the model. Moreover, features with correlation coefficient close to 1 can be redundant if they share the same information. However, the correlation coefficient only makes sense when the features are linear between each other. If the relation between two features is non-linear, it can produce weird problems. In order to avoid that, we can use the mutual information gain.
"""

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt


def plot_correlation_matrix(X_train):
	fig = plt.subplots(figsize = (12,10))

	corr_matrix = X_train.corr().abs()
	mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

	tri_df = corr_matrix.mask(mask)

	sns.heatmap(corr_matrix, mask=mask, cmap="magma", square=True, vmin=-1, vmax=1, center=0, annot=False)


def plot_residuals(model, X_train, y_train, X_test, y_test):
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)

	plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
	plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Testing data")
	plt.title("Linear regression")
	plt.xlabel("Predicted values")
	plt.ylabel("Residuals")
	plt.legend(loc = "upper left")
	plt.hlines(y = 0, color = "red")
	plt.show()

def plot_predictions(model, X_train, y_train, X_test, y_test):
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)

	plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
	plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Testing data")
	plt.title("Linear regression")
	plt.xlabel("Predicted values")
	plt.ylabel("Real values")
	plt.legend(loc = "upper left")
	plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
	plt.show()