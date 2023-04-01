import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score


if __name__ == "__main__":
   df = pd.read_csv('Churn_Modelling.csv')
   X = df.iloc[:, 3:-1].values
   y = df.iloc[:, -1].values

   le = LabelEncoder()
   X[:, 2] = le.fit_transform(X[:, 2])
   ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
   X = np.array(ct.fit_transform(X))
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   
   ann = tf.keras.models.Sequential()
   ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # hidden layer
   ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # 2nd hidden layer
   ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # output layer sigmoid allows to have the prediction and the probability of the binary outpout to be one

   ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
   
   ann.fit(X_train, y_train, batch_size=32, epochs=100)

   print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5) # trained with scaled value so here we need to use sc.transform for prediction (btw Result is FALSE)

   y_pred = ann.predict(X_test)
   y_pred = (y_pred > 0.5) # On dataset
   # print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
   # Results: left => prediction, right => actual value .. It looks good, let's check with a confusion matrix
   #  [[0 0]
   #  [0 1]
   #  ...
   #  [0 0]]

   cm = confusion_matrix(y_test, y_pred)
   print(cm)
   print(accuracy_score(y_test, y_pred)) # 87% is a pretty descent accuracy
