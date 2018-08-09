# Data Preprocessing Template

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importing the dataset
data = pd.read_csv('train.csv')

# DataFrame Statistics
data.shape #Get dimensions of data set
data.columns #SalePrice is the Y we are trying to predict
data.describe() #Get descriptive statistics of the variables
data.dtypes #What is the data type of the variables
data.isnull().sum().sort_values(ascending=False) #Check Nulls
#Count plots of columns with NA values.
#A fair number of them have significantly skewed distributions. They won't be helpful.
fig, ax = plt.subplots(1,14)
fig.set_size_inches(100,5)
sns.countplot(data["Neighborhood"], ax = ax[0])
sns.countplot(data["Alley"], ax = ax[1])
sns.countplot(data["LotFrontage"], ax = ax[2])
sns.countplot(data["MasVnrType"], ax = ax[3])
sns.countplot(data["MasVnrArea"], ax = ax[4])
sns.countplot(data["BsmtQual"], ax = ax[5])
sns.countplot(data["BsmtCond"], ax = ax[6])
sns.countplot(data["BsmtExposure"], ax = ax[7])
sns.countplot(data["BsmtFinType1"], ax = ax[8])
sns.countplot(data["BsmtFinType2"], ax = ax[9])
sns.countplot(data["Electrical"], ax = ax[10])
sns.countplot(data["FireplaceQu"], ax = ax[11])
sns.countplot(data["GarageType"], ax = ax[12])
sns.countplot(data["GarageYrBlt"], ax = ax[13])
fig.show()
data = data.dropna(axis='columns') #Drop columns with missing values
data = pd.get_dummies(data) #split categorical columns into binary values
#Split X and Y
Y = np.array(data['SalePrice'])
X = data.drop('SalePrice', axis = 1)
X_list = list(X.columns)
X = np.array(X)
X.shape

#How to determine variable importance?
#First solution: Can use PCA to reduce the dimension of the input variable space
#Use PCA project to lower eigenvector space and use this to make predictions
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X[:,1:])
from sklearn.decomposition import PCA 
pca = PCA(n_components=10)
Xpca = pca.fit_transform(X_std)
Xpca.shape
Total = sum(pca.explained_variance_)
percent = pca.explained_variance_/Total
plt.plot(np.cumsum(percent))
plt.plot(6,sum(percent[0:6]),'ro')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative sum of explained variance')
plt.show()
#With 5 principal components we can explain over 75% of the variation in Y
#This is great, we have reduced the number of variables we need to 6.
W = Xpca[:,0:6] #Build new information matrix containing as columns the 1st 6 eigenvectors
z = np.ones((1460,1))
W = np.append(z,W, axis = 1) #Append a matrix of 1s to the beginning to add a y-intercept
X_model = np.append(z,X[:,1:], axis = 1) #X_model to be used for other models later (Ridge & Lasso)

#Linear Regression on first 6 principal components
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
model = linear_model.LinearRegression()
model.fit(W,Y)
model.coef_ #Coefficiants of the model
model.intercept_ #Intercept of the model
model.score(W,Y) #R-squared
#Cross Validation
#10-Fold cv
scores = np.sqrt(-cross_val_score(model,W,Y,scoring="neg_mean_squared_error",cv = 10))
scores.mean() #MSE from 10-fold cv

#Cross-Validation MSE using Ridge Model
model = linear_model.Ridge()
model.fit(X_model,Y)
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75] #Tuning parameters
#Creates an array of MSE estimates based on cv for different level of alpha
cv_ridge = [np.sqrt(-cross_val_score(Ridge(alpha = alpha), X_model, Y,
                            scoring="neg_mean_squared_error", cv = 10)) .mean()
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas) #prepare array to be plotted
cv_ridge.plot(title = "MSE vs. Alpha (Tuning Parameter) - Ridge")
plt.xlabel("alpha")
plt.ylabel("MSE")
#Minimum appears to be reached where alpha = 10
#MSE of model with alpha = 10
cv_ridge.min()
model.score(X_model,Y) #R-squared

#Cross-Validation MSE using Lasso Model
model = linear_model.Lasso()
model.fit(X_model,Y)
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75, 150, 300, 400, 500, 600, 750] #Tuning parameters
#Same idea as for Ridge (above)
#Creates an array of MSE estimates based on cv for different level of alpha
cv_lasso = [np.sqrt(-cross_val_score(Lasso(alpha = alpha), X_model, Y,
                            scoring="neg_mean_squared_error", cv = 10)) .mean()
            for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas) #prepare array to be plotted
cv_lasso.plot(title = "MSE vs. Alpha (Tuning Parameter) - Lasso")
plt.xlabel("alpha")
plt.ylabel("MSE")
plt.show()
cv_lasso.min() #Min MSE
model.score(X_model,Y) #R-squared

#Summary: 
#Columns with NA were dropped
#Tried 3 methods: PCA Regression, Ridge Regression, and Lasso Regression
#Respective MSE estimates based on 10-fold cross validation: 37,361 - 30,350 - 29,711
#lasso performs best
#This is supported by the increase in the R-squared value: 76% - 90% - 92%
