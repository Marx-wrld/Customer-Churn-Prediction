# Customer-Churn-Prediction
ML model that can predict if a customer is going to leave a bank or not. I've used the Random Forest classifier, MLP classifier, and Neural Networks model then got a performance check for each model.

## Installing Required Libraries

``` pip install eli5 ```

## Importing the Required Libraries
```
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
```
## Loading the data
