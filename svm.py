import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


iris = pd.read_csv("iris.csv")

# print(iris.head())
# print(iris.info())
# print(iris.Species.value_counts())


from sklearn.datasets import load_iris #imported iris dataset

iris1 = load_iris() #assigned iris dataset to iris1 var

# corr = iris.corr() #correlation in iris dataset

# sns.heatmap(corr, annot=True) #heatmap of corr

# plt.show()

X = iris1.data[:,[2,3]] # here we only take the petallength and petalwidth features
                        #imported iris dataset contains only feature values in array form so we dont need to remove id and species columns for training purposes.
y = iris1.target

iris_df = pd.DataFrame(X,columns=iris1.feature_names[2:]) #converting x np array to pd dataframe

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=0) #seperating 70% of our feature set and classes for training the ai. remaining 30% will be our test set.
                                                                                      #ai reads the test set for unbiased evaluation of the final model. 

from sklearn.preprocessing import StandardScaler

sc = StandardScaler() #with this method we're scaling our data with unit variance

sc.fit(X_train)

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.svm import SVC #we will train our ai with support vector machine algorithm

svm = SVC(kernel="poly",degree=4,gamma= "auto")
svm.fit(X_train,y_train)

# print(svm.score(X_test,y_test))

from sklearn.neighbors import KNeighborsClassifier #we will train our ai with k-nearest neighbors

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)

# print(knn.score(X_test,y_test))

for c, deger in enumerate (np.unique(y)):
    plt.scatter(x=X[y==deger,0],y=X[y==deger,1])
plt.show()
