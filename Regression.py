#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
import matplotlib.pyplot as plt


# In[ ]:


# number of data points
n_points = 200


# In[ ]:


# set a seed here to initialize the random number generator
# (such that we get the same dataset each time this cell is executed)
numpy.random.seed(1)

# let's generate some "non-linear" data; note
# that the sorting step is done for visualization
# purposes only (to plot the models as connected lines)
X_train = numpy.random.uniform(-10, 10, n_points)
t_train = - X_train**2 + numpy.random.random(n_points) * 25

# generate some points for plotting
X_plot = numpy.arange(X_train.min(), X_train.max(), 0.01)


# reshape all arrays to make sure that we deal with
# N-dimensional Numpy arrays
t_train = t_train.reshape((len(t_train), 1))
X_train = X_train.reshape((len(X_train), 1))
X_plot = X_plot.reshape((len(X_plot), 1))
print(X_plot.shape)

print("Shape of training data: %s" % str(X_train.shape))
print("Shape of target vector: %s" % str(t_train.shape))
print("Shape of plotting data: %s" % str(X_plot.shape))

# print(X_train)
# In[ ]:


import linreg

# instantiate the regression model
model = linreg.LinearRegression()

# fit the model
model.fit(X_train, t_train)


# In[ ]:
# get predictions for the data points
preds = model.predict(X_plot)
print(model.w)
# plot the points and the linear regression model
plt.plot(X_train, t_train, 'o')
plt.plot(X_plot, preds, '-', color='red')



# In[ ]:


# 4 b)

# When Sig =0.1
model.fitband(X_train, t_train, 0.1)
print('sig=.1\n', model.w1)
preds_1 = model.predict_b(X_plot)

plt.figure()
plt.plot(X_train, t_train, 'o')
plt.plot(X_plot, preds_1, '-', color='blue')
plt.title('when sigma =0.1')

# when sig=1
model.fitband(X_train, t_train, 1)
print('sig=1\n', model.w1)
preds_2 = model.predict_b(X_plot)

plt.figure()
plt.plot(X_train, t_train, 'o')
plt.plot(X_plot, preds_2, '-', color='green')
plt.title('when sigma =1')

# when sig=10
model.fitband(X_train, t_train, 10)
print('sig=10\n', model.w1)
preds_3 = model.predict_b(X_plot)

plt.figure()
plt.plot(X_train, t_train, 'o')
plt.plot(X_plot, preds_3, '-', color='black')
plt.title('when sigma =10')
plt.show()

# TODO: Implement the non-linear regression approach;
# generate corresponding plots for sigma=0.1,
# sigma=1.0, and sigma=10.0 by computing, for each
# xbar in X_plot, the corresponding prediction


# In[ ]:
