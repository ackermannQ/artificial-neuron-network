import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap


def load_dataframe(path):
   df = pd.read_csv(path)
   X = df.iloc[:, :-1].values
   y = df.iloc[:, -1].values

   return X, y

def plot_set(X, y, sc):
   X_set, y_set = sc.inverse_transform(X), y
   X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                        np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
   plt.contourf(X1, X2, logisticRegression.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
               alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
   plt.xlim(X1.min(), X1.max())
   plt.ylim(X2.min(), X2.max())
   for i, j in enumerate(np.unique(y_set)):
      plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
   plt.title('Logistic Regression')
   plt.xlabel('Age')
   plt.ylabel('Estimated Salary')
   plt.legend()
   plt.show()


if __name__ == "__main__":
   # Load dataframe
   X, y = load_dataframe('Social_Network_Ads.csv')
   
   # Build test and train sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

   # Standardize
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.fit_transform(X_test)

   # Build and train logistic regression model
   logisticRegression = LogisticRegression(random_state = 0)
   logisticRegression.fit(X_train, y_train)

   # Use the logistic regression to predict y_pred and display it by resizing the arrays
   y_pred = logisticRegression.predict(X_test)
   # print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

   cm = confusion_matrix(y_test, y_pred)
   print(cm)
   print(accuracy_score(y_test, y_pred)) # 87% precision on the predictions

   plot_set(X_test, y_test, sc)