#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-05T20:39:42.350Z
"""

# # Breast Cancer Prediction Project


# ## import libraries


pip install tensorflow

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn.datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras 
tf.random.set_seed(3)
from keras.models import Sequential
from keras.layers import Flatten, Dense,Input
from sklearn.preprocessing import StandardScaler



data=sklearn.datasets.load_breast_cancer()

# ### Data Uploading


print(data)

df=pd.DataFrame(data.data,columns=data.feature_names)

df['target']=data.target




X = df.drop('target', axis=1)
y = df['target']



print(X.shape,y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# ## standarize the data 


scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# ## Nueral network settups


model = Sequential([
    Input(shape=(30,)), 
    Dense(64, activation='relu'),
    Dense(2, activation='sigmoid')  
])

# ### MODEL COMPILATION


model.compile(optimizer='adam',
              metrics=['accuracy'],
              loss='sparse_categorical_crossentropy')

history=model.fit(X_train_scaled,y_train,validation_split=.1,epochs=10)


# ### Building a Predictive System


first_row = df.iloc[1].values
print(first_row)


input_data = (
    2.057e+01, 1.777e+01, 1.329e+02, 1.326e+03, 8.474e-02, 7.864e-02, 8.690e-02,
    7.017e-02, 1.812e-01, 5.667e-02, 5.435e-01, 7.339e-01, 3.398e+00, 7.408e+01,
    5.225e-03, 1.308e-02, 1.860e-02, 1.340e-02, 1.389e-02, 3.532e-03, 2.499e+01,
    2.341e+01, 1.588e+02, 1.956e+03, 1.238e-01, 1.866e-01, 2.416e-01, 1.860e-01,
    2.750e-01, 8.902e-02
)

input_data_as_array = np.asarray(input_data)

input_data_reshaped = input_data_as_array.reshape(1, -1)

input_data_scaled = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_scaled)
print("Raw prediction output:", prediction)

if prediction.shape[1] == 1:
    prediction_label = int(prediction[0][0] > 0.5)

else:
    prediction_label = int(np.argmax(prediction))

print("Predicted label:", prediction_label)

if prediction_label == 0:
    print("the tumor is MALIGNANT")
else:
    print(" The tumor is BENIGN")
