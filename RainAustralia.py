# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:54:41 2021

@author: Alan
"""

# Parte 1 - Pre procesado de datos

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('weatherAUS.csv')
dataset=dataset.dropna(subset=["RainTomorrow"])
dataset=dataset.drop(['Date','Evaporation','Sunshine','Cloud9am','Cloud3pm'], axis=1)


X = dataset.iloc[:, 0:17].values
y = dataset.iloc[:, [17]].values


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder() #Location
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])

labelencoder_X_4 = LabelEncoder() #WindGustDir
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])

labelencoder_X_6 = LabelEncoder() #WindDir9am
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])

labelencoder_X_7 = LabelEncoder() #WindDir3pm
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])

labelencoder_X_16 = LabelEncoder() #WindDir3pm
X[:, 16] = labelencoder_X_16.fit_transform(X[:, 16])

labelencoder_y_0 = LabelEncoder() 
y = labelencoder_y_0.fit_transform(y)

y= y.reshape((142193, 1))


onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X=X.toarray()
X = X[:, 1:]

onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [51])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [68])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [85])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [135])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]


#Tratamineto de NaN
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0)
imputer=imputer.fit(X[:,0:137])
X[:,0:137]=imputer.transform(X[:,0:137])
y.reshape(1, -1)
imputer=imputer.fit(y)
y=imputer.transform(y)

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential #Inicializar los parametros de la red neuronal
from keras.layers import Dense

classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 54, kernel_initializer = "uniform",  
                     activation = "relu", input_dim = 137))

# Añadir la segunda capa oculta
classifier.add(Dense(units = 54, kernel_initializer = "uniform",  activation = "relu"))

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train,  batch_size = 10, epochs = 100)


# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(20343+3564)/28439





















