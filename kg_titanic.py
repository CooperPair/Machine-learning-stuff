import pandas as pd
import numpy as np
import random as rnd
import os
import re
import itertools
import csv
#Supervised ML Models
from sklearn.ensemble import RandomForestClassifier
#using unsupervised model
from sklearn.decomposition import PCA

# Evalaluation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# Grid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st
# Warnings
import warnings
warnings.filterwarnings(action='ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

#Master Parameter
n_splits = 5
n_iter = 100
scoring = 'accuracy'
rstate = 45
testset_size = 0.20
num_rounds = 1000
n_tree_range = st.randint(100, num_rounds)

#Load
train_df = pd.read_csv("train.csv", index_col='PassengerId')
test_df = pd.read_csv("test.csv", index_col='PassengerId')

Survived = train_df['Survived'].copy()
train_df = train_df.drop('Survived' ,axis=1).copy()
df = pd.concat([train_df, test_df])

traindex = train_df.index

testdex = test_df.index

del train_df
del test_df

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df["Name_length"] = df['Name'].apply(len)

df['IsAlone'] = 0

df.loc[df['FamilySize']==1, 'IsAlone'] = 1

df['Title'] = 0
df['Title'] = df.Name.str.extract('([A-Za-z]+)\.') #extracting the Saluations

df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
			['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

df.loc[(df.Age.isnull())&(df.Title=='Mr'), 'Age'] = df.Age[df.Title=='Mr'].mean()

df.loc[(df.Age.isnull())&(df.Title=='Mrs'), 'Age'] = df.Age[df.Title=='Mrs'].mean()

df.loc[(df.Age.isnull())&(df.Title=='Master'), 'Age'] = df.Age[df.Title=='Master'].mean()

df.loc[(df.Age.isnull())&(df.Title=='Miss'), 'Age'] = df.Age[df.Title=='Miss'].mean()

df.loc[(df.Age.isnull())&(df.Title=='Other'), 'Age'] = df.Age[df.Title=='Other'].mean()

df = df.drop('Name', axis = 1)

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

#Continuous Variable
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

df['Sex'] = df['Sex'].map({'female':1,'male':0}).astype(int)

df['Title'] = df['Title'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Other':4})

df['Title'] = df['Title'].fillna(df['Title'].mode().iloc[0])#if not given
df['Title'] = df['Title'].astype(int)

df['Embarked'] = df['Embarked'].map({'Q':0, 'S': 1, 'C':2}).astype(int)

df = df.drop(['Ticket', 'Cabin'], axis=1)

categorical_features = ['Pclass', "Sex", "IsAlone","Title", "Embarked"]

from sklearn import preprocessing
continuous_features = ['Fare', 'Age', 'Name_length']
for col in continuous_features:
	transf = df[col].values.reshape(-1,1)
	scaler = preprocessing.StandardScaler().fit(transf)
	df[col] = scaler.transform(transf)

#now that pre-processing task is complete, split the data into training and test set again.
train_df = df.loc[traindex, :]
train_df['Survived'] = Survived
test_df = df.loc[testdex, :]
del df

#dependent and independent variable..
X = train_df.drop(["Survived"] ,axis =1)
y = train_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testset_size, stratify=y,
															random_state=rstate)
#Stratified Cross-Validation
cv = StratifiedShuffleSplit(n_splits, test_size=0.2, random_state=rstate)

def predict_result(model):
	model.fit(X, y)
	submission = model.predict(test_df)
	df = pd.DataFrame({'PassengerId':test_df.index,'Survived':submission})
	print(df)

model = RandomForestClassifier()

param_grid ={'max_depth': st.randint(6, 11),'n_estimators': n_tree_range,
			'max_features':np.arange(0.5,.81, 0.05),'max_leaf_nodes':st.randint(6, 10)}

grid = RandomizedSearchCV(model,param_grid, cv=cv,scoring=scoring,verbose=1,n_iter=n_iter,random_state=rstate)
grid.fit(X_train, y_train)
predict_result(grid)