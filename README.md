# movie-revenue-predictor
Project for LELEC2870 course. Goal is to apply Machine Learning techniques to predict movie's revenue in the USA.    

## Datasets
Datasets can be downloaded from this link: https://drive.google.com/drive/folders/1J2DbMYFF1PCmD3pTDfgHP4tLvxlB1Dz- and should be put in a `datasets` folder at the root of this folder. 

## Packages
```
$ conda env create -n <envname> --file environment.yml
$ conda activate <envname>
$ pip install jupyter (or with anaconda explorer)
$ pip install ipywidgets widgetsnbextension pandas-profiling
$ jupyter nbextension enable --py widgetsnbextension
```

The models implementations are available in the `movie-revenue-prediction.ipynb` file.