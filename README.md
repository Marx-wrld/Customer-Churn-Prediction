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
```
data = pd.read_csv('Churn_Modelling.csv')
```
## Choosing the Features
```
X = data.iloc[:,3:-1]
```
## Encoding the Categorical Features

- Our columns e.g Geography contains words and not numbers, These words can assume a limited number of values, called Categorical features
- Our ML model is mathematical so, it requires numbers for computation that's why we encode
```
encoder = OrdinalEncoder()
value = encoder.fit_transform(X['Geography'].values.reshape(-1, 1))
X['Geography'] = value
encoder = OrdinalEncoder()
value = encoder.fit_transform(X['Gender'].values.reshape(-1, 1))
X['Gender'] = value
```
- We've encoded the Geography and Gender columns with OrdinalEncoder()

## Getting the Target Column
- Predicting whether a customer leaves the bank is a supervised learning problem and so we have to train the model so as to be able to predict the right target variable which is a column of 0s and 1s.
```
y = data.iloc[:, len(data.columns)-1]
```
