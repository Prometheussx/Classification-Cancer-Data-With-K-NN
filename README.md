# 🦠 Cancer Data Classification With K-NN
### 🏆 Test accuracy: 95.32% (3 nn)
## 📑 Project Summary
Using the Knn algorithm, we analyze and classify the cancer cells we have as benign or malignant

*1 = M (Malignant Cancer Cell)*

*0 = B (Benign Cancer Cell)*
## ⚙️ Data information
My data consists of 569 cancer cells and 30 characteristics of each cell.

## 📈 Plot of  Find K Values
```python
from sklearn.neighbors import KNeighborsClassifier
Score_list = []

for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    Score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),Score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
```
![__results___21_0 (1)](https://github.com/Prometheussx/Classification-Cancer-Data-With-K-NN/assets/54312783/decd61a3-69fb-48e9-b9f2-af0ba9cfeeaa)

## 🤖 K-NN Model
Here we examine the accuracy score of the K-nn Model and we get a score of 0.953216373742690059 which indicates that we have trained a good model in general, of course, better results can be obtained by playing with the values.
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("{} nn {} score".format(3,knn.score(x_test,y_test)))
```
## Result
🏆 Test accuracy: 95.32% (3 nn)
