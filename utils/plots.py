""" 
A feature perfectly correlated with another one does not apport any new informations but add complexity to the model. Moreover, features with correlation coefficient close to 1 can be redundant if they share the same information. However, the correlation coefficient only makes sense when the features are linear between each other. If the relation between two features is non-linear, it can produce weird problems. In order to avoid that, we can use the mutual information gain.
"""

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_predict, validation_curve, learning_curve

from utils.feature_selection import get_discrete_features_mask
from utils.latex import set_size, set_size_square_plot

import random


def plot_correlation_matrix(X_train):
    fig = plt.subplots(figsize=(12, 10))

    corr_matrix = X_train.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    tri_df = corr_matrix.mask(mask)

    sns.heatmap(corr_matrix, mask=mask, cmap="magma", square=True,
                vmin=-1, vmax=1, center=0, annot=False)


def plot_mutual_information_matrix(X_train, MI_matrix):
    fig = plt.subplots(figsize=(set_size_square_plot(width="full-size")))

    MI_df = pd.DataFrame(
        MI_matrix, columns=X_train.columns, index=X_train.columns)

    mask = np.triu(np.ones_like(MI_df, dtype=bool))

    sns.heatmap(MI_df, mask=mask, cmap=plt.cm.Reds, vmin=0, vmax=1, center=0)


def plot_mutual_information_with_target(X_train, y_train):
    fig = plt.subplots(figsize=(set_size_square_plot(width="full-size")))

    MI_with_target = mutual_info_regression(X_train, y_train)
    idx_sorted = np.argsort(MI_with_target)[::-1]

    MI = MI_with_target[idx_sorted]
    columns = X_train.columns[idx_sorted].values

    sns.barplot(x=MI, y=columns)


def plot_residuals(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    plt.scatter(y_train_pred, y_train_pred - y_train,
                c="blue", marker="s", label="Training data")
    plt.scatter(y_test_pred, y_test_pred - y_test,
                c="lightgreen", marker="s", label="Testing data")
    plt.title("Linear regression")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, color="red")
    plt.show()


def plot_predictions(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    plt.scatter(y_train_pred, y_train, c="blue",
                marker="s", label="Training data")
    plt.scatter(y_test_pred, y_test, c="lightgreen",
                marker="s", label="Testing data")
    plt.title("Linear regression")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()

# to know how many data you need


def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, kf, scorer):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, train_sizes=np.linspace(0.2, 1.0, 5), cv=kf, scoring=scorer)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(12, 8))

    plt.plot(train_sizes, train_mean, color="blue",
             marker="o", markersize=5, label="train score")
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, alpha=0.15, color="blue")

    plt.plot(train_sizes, val_mean, color="green", linestyle="--",
             marker="s", markersize=5, label="validation score")
    plt.fill_between(train_sizes, val_mean + val_std,
                     val_mean - val_std, alpha=0.15, color="green")

    plt.title(f"learning curve: {model_name}")
    plt.xlabel("train sizes")
    plt.ylabel("score (RMSE)")
    plt.legend()

# detect overfitting


def validate_model(model, model_name, param_name, param_range, X_train, y_train, X_test, y_test, kf, scorer):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    train_scores, val_scores = validation_curve(
        model, X_train, y_train, param_name=param_name, param_range=param_range, cv=kf, scoring=scorer)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(12, 8))

    plt.plot(param_range, train_mean, color="blue",
             marker="o", markersize=5, label="train score")
    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15, color="blue")

    plt.plot(param_range, val_mean, color="green", linestyle="--",
             marker="s", markersize=5, label="validation score")
    plt.fill_between(param_range, val_mean + val_std,
                     val_mean - val_std, alpha=0.15, color="green")

    plt.title(f"validation curve: {model_name}")
    plt.xlabel(param_name)
    plt.ylabel("score (RMSE)")
    plt.legend()


def generate_color_palette(n):
    return list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n)))


def validate_model_with_feature_selection(percentiles, models, model_name, param_name, param_range, X_train, y_train, X_test, y_test, kf, scorer):
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = generate_color_palette(len(percentiles))

    for percentile, model, color in zip(percentiles, models, colors):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        train_scores, val_scores = validation_curve(
            model, X_train, y_train, param_name=param_name, param_range=param_range, cv=kf, scoring=scorer)

        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)

        ax.plot(param_range, train_mean, color=color, marker="o",
                markersize=5, label=f"{percentile}% features kept")

        ax.plot(param_range, val_mean, color="green",
                linestyle="--", marker="s", markersize=5)

    plt.title(f"validation curve: {model_name}")
    plt.xlabel(param_name)
    plt.ylabel("RMSE")
    plt.legend()
