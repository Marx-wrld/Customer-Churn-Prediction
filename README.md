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
## Splitting the Data into Train and Test sets
- The function train_test_split will be used to divide our data into training and testing sets.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
```
## Checking the lengths of train and test sets in which the general practice requires 70% to be training and 30% to be testing
```
len(X_train), len(X_test), len(y_train), len(y_test)
```
## Implementing the Random Forest Classifier Model
- Trying the model for customers who will leave the bank.
```
RF = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 0)

RF.fit(X_train, y_train)
```
## Saving the trained model to a file
```
model_filename = 'random_forest_model.pkl'
joblib.dump(RF, model_filename)
```
## Downloading the model
```
import shutil
shutil.move('random_forest_model.pkl', 'random_forest_model_download.pkl')
```
## Performance Check for Random Forest classifier
```
round(RF.score(X_train, y_train), 4)
```
- Training we get an 80.93% accuracy
```
round(RF.score(X_test, y_test), 4)
```
- Testing we get an 82.37% accuracy

## Checking Feature Importance for Random Forest Classifier Model

- Here is where we use eli5 to get feature importance
```
perm = PermutationImportance(RF, random_state = 42, n_iter = 10).fit(X, y)

eli5.show_weights(perm, feature_names = X.columns.tolist())
```
- This now indicates that NumOfProducts(age and balance) are our Top features
