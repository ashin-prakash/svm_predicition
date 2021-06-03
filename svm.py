import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("finalsample.csv")
print(df.head())

df.columns = ['Age', 'is_male', 'S1', 'S2', 'split_btw_S1_S2', 'split_btw_S2_S1', 'type_of_murmur', 'Result']

print(df.describe())

a = pd.get_dummies(df['type_of_murmur'], prefix = "mur")

frames = [df, a]
df = pd.concat(frames, axis = 1)
print(df.columns)

# dropping categorized columns / thal_0 miscreated by get_dummies
df = df.drop(columns = ['type_of_murmur'])

# renaming categorical columns
df = df.rename(columns={'mur_0.0':'no_Murmur', 'mur_1.0' : 'Systolic_murmur', 'mur_2.0' : 'Diastolic_murmur', 'mur_3.0' : 'Undefined'})
print(df.head())

x_data = df.drop(['Age', 'is_male','split_btw_S1_S2', 'split_btw_S2_S1', 'Result', 'mur_0', 'mur_1', 'mur_2', 'mur_3'], axis=1)

# normalization
print(x_data.head())
X = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

y = df['Result']

from sklearn.svm import SVC


svc=SVC(kernel= 'rbf', gamma = 'scale' ,random_state=42)
svc.fit(x_data, y)
print("Hello")

pickle.dump(svc, open('model.pkl','wb'))
print("Hello")

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[31, 73]]))