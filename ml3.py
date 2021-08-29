#Data source: https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
import os
#Python version
import sys
print('Python: {}'.format(sys.version))
#scipy version
import scipy as sc
print('scipy: {}'.format(sc.__version__))
#numpy version
import numpy as np
print('numpy: {}'.format(np.__version__))
#matplotlib version
import matplotlib.pyplot as plt
#pandas version
import pandas as pd
print('pandas: {}'.format(pd.__version__))
#scikit-learn version
import sklearn as sk
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.svm
import sklearn.metrics
print('sklearn: {}'.format(sk.__version__))
#seaborn version
import seaborn as sns
print('seaborn: {}'.format(sns.__version__))


#Establish Visualization Theme with seaborn
sns.set_theme(style="whitegrid")

#Obtain working directory
print(os.getcwd())

#Import Data and Show Columns
df = pd.read_csv(r'C:\Users\Admin\Downloads\heart.csv')
print(df.head())

#Column Labels:

#age - Age of the person

#sex - Gender of the person (1 = men, 0 = women)

#cp - Chest Pain type

#trtbps - resting blood pressure (in mm Hg)

#chol - cholestoral in mg/dl fetched via BMI sensor

#fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

#restecg - resting electrocardiographic results

#thalachh - maximum heart rate achieved

#exng - exercise induced angina (1 = yes; 0 = no)

#oldpeak - Previous peak

#slp - slope

#caa - number of major vessels (0-3)

#thall - thal rate

#output - target variable (1 = more likely to have a heart attack; 0 = less likely to have a heart attack)

#Statistics summary of continuous variables
num_df = df[["age","chol","thalachh","oldpeak","trtbps"]]
print(num_df.describe())

#Statistics summary for discrete variables
print(df['sex'].value_counts())
print(df['output'].value_counts())
print(df['fbs'].value_counts())
print(df['slp'].value_counts())
print(df['caa'].value_counts())
print(df['thall'].value_counts())
print(df['cp'].value_counts())
print(df['restecg'].value_counts())
print(df['exng'].value_counts())

#Visualizations of discrete variables


sex = df['sex'].value_counts()
sex.plot(kind='bar')
plt.show()

#There are more men than women being represented in the data set

sex = df['output'].value_counts()
sex.plot(kind='bar')
plt.show()
#There are more individuals who are prone to heart attack than not

#Visualizations of continuous variables

sns.histplot(data = df, x = 'age')
plt.show()

sns.histplot(data = df, x = 'chol')
plt.show()

sns.histplot(data = df, x = 'trtbps')
plt.show()

sns.histplot(data = df, x = 'thalachh')
plt.show()


sns.scatterplot(data = df, x="age", y="chol")
plt.show()

sns.scatterplot(data = df, x='trtbps', y='thalachh')
plt.show()

#Development of machine learning classification model

#Converting data to numpy array for input to machine learning algorithm,
#where X constitutes features and y is output (target variable)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Splitting data into training and testing data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Standardizing features to ensure they're all on the same scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the SVM model on the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, C = 2)
classifier.fit(X_train, y_train)

#Predicting the results from the test set argument
y_pred = classifier.predict(X_test)

#Implementation of the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
#Accuracy Prediction
print(cm)
print(accuracy_score(y_test, y_pred))
#SVM algorithm yields an accuracy score of 86%, which is greater than
#80% and an adequate baseline for most classification issues. SVM was
#chosen over tree-based algorithms as they perform better than those
#types in classification problems with unknown data structures and large
#number of features, due to the kernel trick



