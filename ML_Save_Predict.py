
# coding: utf-8

# In[2]:


#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
#Used Pickle for saving ML model
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle


# In[3]:


# Loading data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values


# In[5]:


#Diving data for cross validation
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[8]:


dataframe.head(5)


# In[9]:


# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)


# In[11]:


# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[12]:


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

