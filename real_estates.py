#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("D:/MachineLearningwithHarry/data1.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


df = housing.drop(['Unnamed: 14','Unnamed: 15','Unnamed: 16','Unnamed: 17', 'Unnamed: 18','Unnamed: 19', 'Unnamed: 17', 'Unnamed: 20','Unnamed: 21', 'Unnamed: 22'], axis=1)
print(df)


# In[8]:


df.describe()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
# to visualize along with the graph


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


df.hist(bins=30, figsize=(20,15))


# ## Train-Test Splitting

# In[12]:


# it can be collected from scikit learn
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42) # it will not allow the data to be interchaged between train and test set
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data)*(test_ratio))
    test_indices =  shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# the problem regarding this function is, if it is shuffled n number of times, it will observe all the values irrespective to train and test set and by pattern matching it will overfit the whole dataset for this reason random.seed is used

    


# In[13]:


# train_set, test_set = split_train_test(df, 0.2)


# In[14]:


# print(f"Rows in train_set: {len(train_set)}\n Rows in test_set:{len(test_set)}")


# In[15]:


import numpy as np
from sklearn.model_selection import train_test_split
train_set, test_set = split_train_test(df, 0.2)
np.random.seed(42)
print(f"Rows in train_set: {len(train_set)}\nRows in test_set:{len(test_set)}")


# In[16]:


# It is used for dividing all the variations of data in a dataset into both train and test sets
from sklearn.model_selection import StratifiedShuffleSplit
split =  StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(df, df['CHAS']):
    strat_train_set = housing.loc[train_index] # We Use DataFrame.loc attribute to access a particular cell in the given Dataframe using the index and column labels.
    strat_test_set = housing.loc[test_index]


# In[17]:


#strat_train_set.describe()
#strat_train_set['CHAS'].value_counts() # 376/28 = 13.43
strat_test_set['CHAS'].value_counts()   # 94/7 = 13. 43 ## This is known as stratified shuffle split i.e. equal ratio of train_set and test_set for  a definite attribute


# In[18]:


376/28


# In[19]:


94/7


# In[20]:


df = strat_train_set.copy()


# # Looking for correlations

# In[21]:


corr_matrix = df.corr()
corr_matrix['MEDV'] # here, we can see the correlation factor of MEDV is 1..which means it is a strong positive corr factor. ..
# here medv, rm ar strong pos...and ptratio and lstat are strong negative...these values play great role in deciding price of houses


# In[22]:


corr_matrix['MEDV'] # here, we can see the correlation factor of MEDV is 1..which means it is a strong positive corr factor. ..
# here medv, rm ar strong pos...and ptratio and lstat are strong negative...these values play great role in deciding price of houses


# In[23]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'LSTAT', 'PTRATIO', 'RM','ZN']
scatter_matrix(df[attributes], figsize=(12, 8))


# In[24]:


housing.plot(kind = 'scatter', x='RM', y= 'MEDV', alpha = 0.8) # here alpha is the radius of plot dot


# **Here we can see that for different values of RM(5, 9) we get same value of MEDV(50).. so there is an error in the data. So in order to find this error in the data we have done this correlation**

# # Trying out attribute combinations

# In[25]:


df['TAXRM']=df['TAX']/df['RM']


# In[26]:


df.head()


# In[27]:


corr_matrix = df.corr()
corr_matrix['MEDV']


# **Here, we have got a good negative correlation value of TAXRM**

# In[28]:


df.plot(kind = 'scatter', x= 'TAXRM', y = 'MEDV', alpha=0.8)


# In[29]:


df=strat_train_set.drop("MEDV", axis=1)
df_labels = strat_train_set["MEDV"].copy() # we know that out of 15 attributes, MEDV is the label. Here, we have dropped MEDV from the main dataframe and paste them in separate dataframe called df_labels


# ## Missing Attributes
# **To take missing attributes we have three options:**
# 1. Get rid of missing data points
# 2. Get rid of whole attribute
# 3. Set the value to some value(0, mean, median)

# In[30]:


# (Option-1) If we have two or three data points missing, we could have got rid of the values. But here, we cannot get rid of missing data points as it have five points missing
# (Option-2) If the attribute has a low correlation factor e.g.(CHAR) we could have ommited that, but RM has a good corrletion factor
# (Option-3) In this situation we should use only this option


# In[31]:


a=df.dropna(subset=['RM'])# here the df dataframe is not changed...if we added inplace==Turue then it would also be changed 
a.shape # the attribute RM is dropped


# In[32]:


df.drop('RM', axis=1).shape # option 2
# no RM column as wel as here, the original df dataframe will be unchanged


# In[33]:


median = df['RM'].median() # compute median for option 3


# In[34]:


median


# In[35]:


df['RM'].fillna(median)
# to be noted that the original housing dataframe is remain unchanged


# In[36]:


df.shape


# In[37]:



df.describe() # before we started filling missing attributes


# In[38]:


housing_1= df.drop(['Unnamed: 14','Unnamed: 15','Unnamed: 16','Unnamed: 17', 'Unnamed: 18','Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22'], axis=1)
housing_1


# In[39]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(housing_1)


# In[40]:


imputer.statistics_ # it calculates median for all columns and replaces all the missing values with the median not just only for RM...as we have seen it has a shape of 15


# In[41]:


X= imputer.transform(housing_1) ## To avoid data leakage during cross-validation, imputer computes the statistics on the train data during the fit , stores it and uses it on the test data, during the transform .


# In[42]:


housing_1_tr= pd.DataFrame(X, columns=housing_1.columns) # transform dataframe is created where there is no missing values


# In[43]:


housing_1_tr.describe() ## here, we can see that the RM column has got all the 505 values...i.e. all the missing attributes are filled


# # ScikitLearn Design

# Three types of of objects:
# 1. Estimators: E.G: Imputer(Fit, TRansformer)
# 2. Transformers
# 3. Predictors

# ## Feature Scaling

# Primarily, two types of scaling method:
# 1. Min-max Scaling (Normalization)
#    (value-min)/(max-min).
#    For this function scikit-learn provides class called MinMaxScaler
# 2. Standardization
#    (value-min)/std.
#    For this function sklearn provides class called StandardScaler

# ## Creating a Pipeline

# In[44]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline= Pipeline([('imputer', SimpleImputer(strategy='median')), 
                      ('std_scaler', StandardScaler())])


# In[45]:


housing_num_tr = my_pipeline.fit_transform(housing_1_tr) # so here, the pandas dataframe is converted into a numpy array....and the input of pipeline is actually an array
housing_num_tr

# after the correlation, combining attributes and missing attributes are done, what we do is imputation..This task of imputation and standardization is actually done in the pipeline by fit_transform method
# So, instead of housing_1_tr we can also use just housing_1(which we get beforehand impute)


# In[46]:


housing_num_tr.shape


# # Selecting Desired model for the real_estates

# In[47]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=LinearRegression()
model = RandomForestRegressor()
# model=DecisionTreeRegressor()
model.fit(housing_num_tr, df_labels)


# In[48]:


some_data = housing_1.iloc[:5]


# In[49]:


some_labels= df_labels.iloc[:5]


# In[50]:


prepared_data=my_pipeline.transform(some_data)


# In[51]:


model.predict(prepared_data)


# In[52]:


list(some_labels)


# ## Evaluating the model

# In[53]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
lin_mse = mean_squared_error(df_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)


# In[54]:


lin_mse # here, for LinearRegression we can see the mean_squared error pf our predictive model is 4.71...which has a good amount of error between some_labels
        # for DecisionTreeRegressor we get the mse as 0. That means it has got overfitting error


# In[55]:


lin_rmse


# ## Using better cross-validation tehnique: Cross Validation

# In[56]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, df_labels, scoring= 'neg_mean_squared_error', cv = 10) #ne_mean_sq_error deermines the utility, and cv means cross val for 10 values
rmse_scores = np.sqrt(-scores)


# In[57]:


rmse_scores # so we can see that there are errors in decision tree regressor is lesser than those of DecisionTreeRegressor


# In[58]:


def print_scores(scores):
    print("Scores", scores)
    print("Mean", scores.mean())
    print("Standard Deviation", scores.std())


# In[59]:


print_scores(rmse_scores)


# In[ ]:




