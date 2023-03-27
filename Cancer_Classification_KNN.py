# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 22:58:56 2023

@author: erdem
"""

#%% Library
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Data İncloud
data = pd.read_csv("Cancer_Data.csv") 

#%% Data manipulation
data.info()

data.drop(["id","Unnamed: 32"],axis = 1, inplace = True) #axis sütun sil anlamında inplace ise kopya oluşturma asıl datayı düzenle

data.tail() #sondan ilk 5 gösterir

#%% Data Graph

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean,M.texture_mean,color = "red", label ="Kotu", alpha = 0.3) #alpha = matlık değeri
plt.scatter(B.radius_mean,B.texture_mean,color = "green", label = "iyi", alpha = 0.3)

plt.xlabel("radius_mean")
plt.ylabel("texture_mean")

plt.legend() #labelları gösterir
plt.show()

#%% diagnosis Change 0 or 1
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis
x_data = data.drop(["diagnosis"],axis = 1)

#%% Data Normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#%% train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

#%% K-NN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) # k=3
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print(" {} nn {} score".format(3,knn.score(x_test, y_test)))


#%% find k value
Score_list = []

for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train, y_train)
    Score_list.append(knn2.score(x_test, y_test))

plt.plot(range(1,15),Score_list)
plt.xlabel("k values")
plt.ylabel("accuracy") #doğruluk skoru demek
plt.show()
















