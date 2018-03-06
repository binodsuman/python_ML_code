
# coding: utf-8

# In[10]:


from sklearn import svm

x = [[1],[4],[7],[13],[10]]
y1 = [16, 34, 52, 88, 70]
y2 = [1, 16, 49, 169, 100]

svm_regression_model = svm.SVR(kernel='poly')
svm_regression_model.fit(x,y1)
print svm_regression_model.predict([5])

svm_regression_model = svm.SVR(kernel='poly')
svm_regression_model.fit(x,y2)
print svm_regression_model.predict([5])

