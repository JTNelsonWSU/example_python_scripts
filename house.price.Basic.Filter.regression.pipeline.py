####Exported from Jupyter Notebook
###Early script for feature engineering housing data to predict sale cost using ML regressions

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load in the packages
from math import sqrt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:






# In[ ]:


####Read the data in to train and test the model 
####Most basic data set, no missing data

house_raw = pd.read_csv("house.train.csv",sep=",")
house_raw_clean = house_raw[["MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold","SalePrice"]] # need to include only features that are numerical
house_raw_clean = house_raw_clean.dropna(axis=1,how="any")
house_raw_clean.head()




# In[ ]:


#View the data and the varying features of the train set
house_raw_clean.describe()
pd.plotting.scatter_matrix(house_raw)


# In[ ]:


####Feature engineering
####Not yet written
##Checking for and removing missing data
#house_raw.iloc[:,0:35].isnull().values.any() #will find any column with missing data in it
#for i in range(36):      # this will loop throu and tell which rows still have missing data
#    house_raw.iloc[:,i].isnull().values.any()  
#house_raw_clean = house_raw.dropna(axis=1,how="any")  # removes all coulmns with missing data
#house_raw_clean.shape


# In[ ]:


#Getting training ad test set
np.random.seed(29)
sample = np.random.uniform(0,1,len(house_raw_clean)) <= .7
house_train = house_raw_clean[sample]
house_test = house_raw_clean[sample == False]
house_test.shape
house_train.shape


# In[ ]:


#Running basic regression without feature engineering
## No Regularization
house_reg = LinearRegression(normalize=True) #This builds the shell of the regression model and also normalizes the data
house_reg = house_reg.fit(house_train.drop("SalePrice",axis=1), house_train["SalePrice"]) #Fits the above shell to the train data (response variable = SalePrice)
# Predict using the model
house_pred = house_reg.predict(house_test.drop("SalePrice", axis=1))  #Fits the test data to the model.


# In[ ]:


## L1 (Lasso) Regularization
house_lasso_reg = Lasso(alpha=1.0, max_iter=100000, normalize=True, tol=0.0001)
house_lasso_reg = house_lasso_reg.fit(house_train.drop("SalePrice",axis=1),house_train["SalePrice"])
# Predict using the model
house_lasso_pred = house_lasso_reg.predict(house_test.drop("SalePrice", axis=1))


# In[ ]:


#L2 (Ridge) Regularization
house_ridge_reg = Ridge(alpha=1.0, max_iter=100000, normalize=True, solver='lsqr', tol=0.001)
house_ridge_reg = house_ridge_reg.fit(house_train.drop("SalePrice", axis=1),house_train["SalePrice"])
#Predict using the model
house_ridge_pred = house_ridge_reg.predict(house_test.drop("SalePrice", axis=1))

#####################
###Cross-Validation Ridge
house_ridge_CV_reg = RidgeCV(alphas=(0.01,0.1,1.0,10.0), normalize=True, cv=10)
house_ridge_CV_reg = house_ridge_CV_reg.fit(house_train.drop("SalePrice", axis=1),house_train["SalePrice"])
# Predict using the model
house_ridge_CV_reg_pred = house_ridge_CV_reg.predict(house_test.drop("SalePrice", axis=1))




# In[ ]:


#Comparing the linear coefficients 
print("Linear Coef: " + str(house_reg.coef_)
     + "\nLasso Coef: " + str(house_lasso_reg.coef_)
     + "\nRidge Coef: " + str(house_ridge_reg.coef_)
    + "\nRidgeCV: " + str(house_ridge_CV_reg.coef_)) 
    


# In[ ]:


#Model evaluation
#basic linear model
house_reg_mae = metrics.mean_absolute_error(house_test["SalePrice"], house_pred)
house_reg_rmse = sqrt(metrics.mean_squared_error(house_test["SalePrice"],house_pred))
house_reg_r2 = metrics.r2_score(house_test["SalePrice"],house_pred)

#Lasso linear model
house_lasso_reg_mae = metrics.mean_absolute_error(house_test["SalePrice"], house_lasso_pred)
house_lasso_reg_rmse = sqrt(metrics.mean_squared_error(house_test["SalePrice"],house_lasso_pred))
house_lasso_reg_r2 = metrics.r2_score(house_test["SalePrice"],house_lasso_pred)


#Ridge linear model
house_ridge_reg_mae = metrics.mean_absolute_error(house_test["SalePrice"], house_ridge_pred)
house_ridge_reg_rmse = sqrt(metrics.mean_squared_error(house_test["SalePrice"],house_ridge_pred))
house_ridge_reg_r2 = metrics.r2_score(house_test["SalePrice"],house_ridge_pred)

#RidgeCV linear model
house_ridgeCV_reg_mae = metrics.mean_absolute_error(house_test["SalePrice"],house_ridge_CV_reg_pred)
house_ridgeCV_reg_rmse = sqrt(metrics.mean_squared_error(house_test["SalePrice"],house_ridge_CV_reg_pred))
house_ridgeCV_reg_r2 = metrics.r2_score(house_test["SalePrice"], house_ridge_CV_reg_pred)


# In[ ]:


####Print statisitcs
stats = pd.DataFrame({"Linear":{"MAE":house_reg_mae, "RMSE":house_reg_rmse,"R2":house_reg_r2},
                    "Lasso":{"MAE":house_lasso_reg_mae, "RMSE":house_lasso_reg_rmse,"R2":house_lasso_reg_r2},
                      "Rigde":{"MAE":house_ridge_reg_mae, "RMSE":house_ridge_reg_rmse,"R2":house_ridge_reg_r2},
                         "RidgeCV":{"MAE":house_ridgeCV_reg_mae, "RMSE":house_ridgeCV_reg_rmse, "R2":house_ridgeCV_reg_r2}})

print(stats)


# In[ ]:


########


# In[ ]:


########


# In[ ]:


###########Apply the developed model to a novel data set

house_new_data = pd.read_csv("house.test.csv", sep=",")
house_new_data_clean = house_new_data[["MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold"]] # need to include only features that are numerical
house_new_data_clean = house_new_data_clean.dropna(axis=1,how="any")
house_new_data_clean.head()


# In[ ]:


###Predict the SalePrice
house_reg_final_predict = house_reg.predict(house_new_data_clean)
house_lasso_reg_final_predict = house_lasso_reg.predict(house_new_data_clean)
house_ridge_reg_final_predict = house_ridge_reg.predict(house_new_data_clean)
house_ridge_CV_reg_final_predict = house_ridge_CV_reg.predict(house_new_data_clean)


# In[ ]:


#####Formatting the predicted values for an outfile
reg_ID = pd.DataFrame(data=house_new_data["Id"])
lasso_ID = pd.DataFrame(data=house_new_data["Id"])
ridge_ID = pd.DataFrame(data=house_new_data["Id"])
ridgeCV_ID = pd.DataFrame(data=house_new_data["Id"])

reg_out = pd.DataFrame(data=house_reg_final_predict)
lasso_out = pd.DataFrame(data=house_lasso_reg_final_predict)
ridge_out = pd.DataFrame(data=house_ridge_reg_final_predict)
ridgeCV_out = pd.DataFrame(data=house_ridge_CV_reg_final_predict)

reg_ID["SalePrice"] = reg_out[0]
reg_ID.to_csv("reg.basic.predictions.txt",index=False)

lasso_ID["SalePrice"] = lasso_out[0]
lasso_ID.to_csv("lasso.basic.predictions.txt",index=False)

ridge_ID["SalePrice"] = ridge_out[0]
ridge_ID.to_csv("ridge.basic.predictions.txt",index=False)

ridgeCV_ID["SalePrice"] = ridgeCV_out[0]
ridgeCV_ID.to_csv("ridgeCV.basic.predictions.txt",index=False)



# In[ ]:





# In[ ]:





# In[ ]:







# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




