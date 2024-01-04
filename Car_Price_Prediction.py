#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction using ML 

# ## Import Necessary Libraries

# In[52]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')


# ## Loading Dataset

# In[53]:


df=pd.read_csv("C:/Users/Effat/Desktop/SystemTronInternship/CarPricePrediction/quikr_car.csv")


# In[54]:


df.shape


# ## Top 5 row's

# In[55]:


df.head()


# # Last 5 row's

# In[56]:


df.tail()


# # EDA- Exploratory Data Analysis

# ### 1. Information of dataset

# In[57]:


df.info()


# ### 2. Statistic of dataset

# In[58]:


df.describe().T


# ### 3. Null values  

# In[59]:


df.isnull().sum()


# ### 4. Unique values of each Column

# In[60]:


# find unique values of each column
for i in df.columns:
    print("Unique value of:>>> {} ({})\n{}\n".format(i, len(df[i].unique()), df[i].unique()))


# In[61]:


df_copy=df.copy()


# # Quality of dataset as seen from the file directly
# 
# * names are pretty inconsistent
# 
# * names have company names attached to it
# 
# * some names are spam like 'Maruti Ertiga showroom condition with' and 'Well mentained Tata Sumo'
# 
# * company: many of the names are not of any company like 'Used', 'URJENT', and so on.
# 
# * year has many non-year values
# 
# * year is in object. Change to integer
# 
# * Price has Ask for Price
# 
# * Price has commas in its prices and is in object
# 
# * kms_driven has object values with kms at last.
# 
# * It has nan values and two rows have 'Petrol' in them
# 
# * fuel_type has nan values

# # Data Cleaning

# ### 1. Year 

# In[62]:


#year has many non-numeric values
df=df[df['year'].str.isnumeric()]
df.shape


# In[63]:


#year column is an object, change it to integer
df['year']=df['year'].astype(int)


# ### 2. Price

# In[64]:


#price column as 'Ask for price' value in it
df=df[df['Price']!='Ask For Price']
df.shape


# In[65]:


#prices have commas in its rows and is an object to be converted to integer type
df['Price']=df['Price'].str.replace(',','').astype(int)


# ### 3. kms_driven 

# In[66]:


#kms_driven is an object column with kms at last and commas in between
df['kms_driven']=df['kms_driven'].str.split().str.get(0).str.replace(',','')
df.shape


# In[67]:


#kms_driven has nan values and has two petrol values in it
df=df[df['kms_driven'].str.isnumeric()]
df.shape


# In[68]:


df['kms_driven']=df['kms_driven'].astype(int)


# In[69]:


df=df[~df['kms_driven'].isna()]
df.shape


# * Changing car names. Keeping only the first three words

# In[70]:


df['name']=df['name'].str.split().str.slice(start=0,stop=3).str.join('')
df['name'].head()


# Reset the index

# In[71]:


df=df.reset_index(drop=True)


# In[72]:


df


# ## Cleaned Dataset

# In[73]:


df.to_csv("df_cleaned.csv")


# In[74]:


df.info()


# In[75]:


df.describe(include='all')


# In[76]:


df[df['Price']>6000000]


# Since, car which has a price more than 60 lakhs is only one, it can be considered as an outlier.

# In[77]:


df=df[df['Price']<6000000]
df.shape


# # Data Visualization

# ### 1. Realationship between Company name vs Price

# In[78]:


df['company'].unique


# In[79]:


plt.subplots(figsize=(15,7))
ax=sns.barplot(x='company',y='Price',data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# ### 2. Variation of Price with Year

# In[80]:


import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,8))
sns.swarmplot(x='year',y='Price',data=df)
plt.xticks(rotation=45)
plt.show()


# ### 3. Relation with kms_driven with price

# In[81]:


sns.relplot(x='kms_driven',y='Price',data=df,height=6)
plt.show()


# ### 4. Relation with Fuel_type with Price

# In[82]:


plt.figure(figsize=(16,8))
sns.boxplot(x='fuel_type',y='Price',data=df)
plt.show()


# ### 5. Relation of price with company, fuel_type and year all together

# In[83]:


sns.relplot(x='company',y='Price',data=df,hue='fuel_type',size='year',height=7,aspect=2)
plt.xticks(rotation=45)
plt.show()


# ### 6. Correlation of variables

# In[84]:


plt.figure(figsize=(24,13))

d = df
corr = d.corr()
sns.heatmap(corr, annot=True, fmt=".2f");


# # Model Building and Evaluation

# ## 1. Training Data Extraction 

# In[85]:


X=df.drop('Price',axis=1)
y=df['Price']


# In[86]:


X


# In[87]:


y


# ## 2. Creating a pipeline and transformer

# In[88]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder


# In[89]:


df_num=X.select_dtypes(include='number')
df_cat=X.select_dtypes(include='object')


# In[90]:


num_attributes=list(df_num.keys())
cat_attributes=list(df_cat.keys())


# In[91]:


encoder=OneHotEncoder()
encoder.fit(X[cat_attributes])
encoder.categories_


# In[92]:


num_pipeline=Pipeline(steps=[('scaling',StandardScaler())])
cat_pipeline=Pipeline(steps=[('onehotencoding',OneHotEncoder(categories=encoder.categories_,drop='first'))])


# Creating a separate pieline for both categorical and numerical features using column transformer

# In[93]:


transformer=ColumnTransformer(transformers=[('numerical',num_pipeline,num_attributes),
                                            ('categorical',cat_pipeline,cat_attributes)])


# ## 3. Splitting of Train and Test Data

# In[94]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)


# ## 4. Using Random forest Regressor for regression

# In[95]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# In[106]:


rfr=Pipeline([('column_transformer',transformer),
             ('random_forest',RandomForestRegressor())])
rfr.fit(X_train,y_train)


# In[107]:


y_pred=rfr.predict(X_test)


# In[108]:


r2_score(y_test,y_pred)


# In[294]:


r2_score(y_test,y_pred)


# ###  Evaluation

# In[109]:


rfr.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['MarutiSuzukiA','Maruti',2019,100,'Petrol']).reshape(1,5)))


# ## 5. Using Random forest Regression as the final model

# In[316]:


import pickle


# In[317]:


pickle.dump(rfr,open('RandomforestModel.pkl','wb'))


# # Thank You
