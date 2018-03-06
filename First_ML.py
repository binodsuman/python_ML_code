
# coding: utf-8

# In[30]:


#https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
import sys 
sys.version
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



# In[38]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
dataset.head()
#dataset.head(10)
#dataset


# In[40]:


print(dataset.shape)


# In[42]:


print(dataset.describe())


# In[46]:


print(dataset.groupby('class').size())


# In[50]:


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[52]:


dataset.hist()
plt.show()


# In[54]:


scatter_matrix(dataset)
plt.show()


# In[58]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state=seed)


# In[60]:


# Test option and evaluatioin metric.
# We will use 10-fold cross validation to estimate accuracy.
seed = 7
scoring = 'accuracy'


# In[65]:


# Building Model
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(KNeighborsClassifier(), X_train, Y_train, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ('KNN', cv_results.mean(), cv_results.std())
print(msg)


# In[66]:


#Making predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

