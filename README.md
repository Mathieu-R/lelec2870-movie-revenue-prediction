# movie-revenue-predictor
Project for LELEC2870 course. Goal is to apply Machine Learning techniques to predict movie's revenue in the USA.    

## Datasets
Datasets can be downloaded from this link: https://drive.google.com/drive/folders/1J2DbMYFF1PCmD3pTDfgHP4tLvxlB1Dz-    

## Libraries
For this project, we use basic datascience python libraries such as `pandas`, `matplotlib`, `sklearn`,... that are already installed with `conda`. 

We also use `category_encoders` package that implement a set of transformers for encoding categorical variables into numeric.
```bash 
$ conda install -c conda-forge category_encoders
```

We also use `skorch` which is an implementation of `pytorch` using a similar syntax as `sklearn`.
You can install this library in `conda` with the following command:

```bash
$ conda install pytorch cudatoolkit==11.1 -c pytorch # pytorch
$ conda install -c conda-forge skorch # skorch
```

We also use `pandera` for data type validation in our `utils.py`. 
You can install this library in `conda` with the following command:

```bash
$ conda install -c conda-forge pandera
```