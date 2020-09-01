project_name
==============================

An End to End Machine Learning Project: 

The objective is to learn the surrounding infrastructure necessary for reproducible and documented machine learning project. 

The dataset used is the MNIST dataset of 60,000 handwritten digits. A tensorflow, sequential CNN classifier was built which had a test accuracy of 99.26%. 

There is a Tkinter GUI to draw a digit and test the classifier. Due to containerization, the use of Win32gui was replaced with functions from Tkinter which has affected the models performance. 

Due to be able to use the GUI,  follow the instructions for installation of VcXsrv, setting the display variable and running the container. 

<a target="_blank" href="https://dev.to/darksmile92/run-gui-app-in-linux-docker-container-on-windows-host-4kde">How to run GUI app in linux docker container on windows host</a></small></p>

==============================
Model Summary: 

Trained in  Notebooks/1.0_Zafrin_Dhali_HandwrittenDigit_ModelCreation.ipynb
60000 train samples
10000 test samples

batch_size = 128
num_classes = 10
epochs = 10


Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 57s 956us/step - loss: 0.3642 - accuracy: 0.8862 - val_loss: 0.0596 - val_accuracy: 0.9814
Epoch 2/10
60000/60000 [==============================] - 60s 999us/step - loss: 0.1035 - accuracy: 0.9725 - val_loss: 0.0446 - val_accuracy: 0.9868
Epoch 3/10
60000/60000 [==============================] - 57s 957us/step - loss: 0.0734 - accuracy: 0.9808 - val_loss: 0.0324 - val_accuracy: 0.9907
Epoch 4/10
60000/60000 [==============================] - 57s 957us/step - loss: 0.0611 - accuracy: 0.9839 - val_loss: 0.0341 - val_accuracy: 0.9894
Epoch 5/10
60000/60000 [==============================] - 57s 952us/step - loss: 0.0521 - accuracy: 0.9870 - val_loss: 0.0282 - val_accuracy: 0.9915
Epoch 6/10
60000/60000 [==============================] - 56s 935us/step - loss: 0.0450 - accuracy: 0.9884 - val_loss: 0.0276 - val_accuracy: 0.9914
Epoch 7/10
60000/60000 [==============================] - 58s 961us/step - loss: 0.0385 - accuracy: 0.9903 - val_loss: 0.0320 - val_accuracy: 0.9921
Epoch 8/10
60000/60000 [==============================] - 57s 951us/step - loss: 0.0355 - accuracy: 0.9909 - val_loss: 0.0265 - val_accuracy: 0.9924
Epoch 9/10
60000/60000 [==============================] - 55s 915us/step - loss: 0.0315 - accuracy: 0.9915 - val_loss: 0.0246 - val_accuracy: 0.9934
Epoch 10/10
60000/60000 [==============================] - 57s 958us/step - loss: 0.0287 - accuracy: 0.9923 - val_loss: 0.0263 - val_accuracy: 0.9926


Test loss: 0.026313767190270114
Test accuracy: 0.9926000237464905

==============================

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
