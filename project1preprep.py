# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import seaborn as sns
from google.colab import files
from datetime import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import tree
from yellowbrick.regressor import ResidualsPlot


files.upload()

df = pd.read_csv('./clean_data.csv')

df['Income by Gender'] = df.groupby('Gender')['Total_income'].transform('mean')
df['Income by Education'] = df.groupby('Education_type')['Total_income'].transform('mean')
df['Income by Income Type'] = df.groupby('Income_type')['Total_income'].transform('mean')
#maybe a aggregation on the average years employed vs Income type or average years emplyed vs income proper?


df

#sns.set(rc={'figure.figsize':(13,9)})

#plot = sns.barplot(data = df, x = 'Target', y = df.groupby('Target')['ID'].transform('count'))
#plot = sns.barplot(data = df, x = 'Gender', y = df.groupby('Gender')['ID'].transform('count'))
#plot = sns.barplot(data = df, x = 'Education_type', y = df.groupby('Education_type')['ID'].transform('count'))
#plot = sns.barplot(data = df, x = 'Family_status', y = df.groupby('Family_status')['ID'].transform('count'))
#plot = sns.barplot(data = df, x = 'Own_property', y = df.groupby('Own_property')['ID'].transform('count'))
#plot = sns.barplot(data = df, x = 'Housing_type', y = df.groupby('Housing_type')['ID'].transform('count'))


le = preprocessing.LabelEncoder()
oh = preprocessing.OneHotEncoder()
df['Education_type'] = le.fit_transform(df['Education_type'])




temp = pd.get_dummies(df, columns = ['Family_status', 'Housing_type'])
df = pd.concat([df,temp], axis =1)

y_data = df['Target']
x_data = df.drop(columns=['Income_type', 'Occupation_type', 'Target', 'Family_status', 'Housing_type', 'ID'])

#scaler = StandardScaler()

#x_data =scaler.fit_transform(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,test_size = .90, random_state = 15)


#reg = linear_model.LinearRegression()

#reg = reg.fit(x_train, y_train)

#reg.score(x_test, y_test)

#visual = ResidualsPlot(reg)
#visual.fit(x_train, y_train)
#visual.score(x_test, y_test)
#visual.poof()

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(x_train, y_train)
#plt.figure(figsize=(100,80))
#name = x_data.columns
#classes = ['not_Risk', 'nRisk']
#tree.plot_tree(clf, max_depth=5, filled=True,class_names=classes, feature_names=name )
#clf.score(x_test, y_test)


#lr = [-0.012793953061021712,0.0013214506726588748, 0.004985837668201931, -0.021151903018850504]
#ct = [0.7631307929969104, 0.7638173704085136, 0.7591829728801922,0.7635885112713126]
#pr = [10, 30, 60, 90]


#plot = sns.lineplot(x=pr,y=lr)

#df