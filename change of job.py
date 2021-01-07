#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[124]:


train = pd.read_csv('C:\\Users\\user\\Downloads\\aug_train\\aug_train.csv')
test = pd.read_csv('C:\\Users\\user\\Downloads\\aug_test.csv')


# In[125]:


data = pd.merge(train,test,how='outer')
data


# In[8]:


data.shape


# In[9]:


train.shape


# In[126]:


missing = data.isnull().sum()
missing


# In[127]:


data.nunique()


# In[13]:


data.info()


# In[128]:


data['city']=data['city'].apply(lambda x: int(x.split('_')[1]))
data['city']


# In[129]:


data_1=data.copy()
print(data_1.company_size.value_counts())
print(data_1.company_type.value_counts())
print(data_1.last_new_job.value_counts())


# In[130]:


data_1['company_type'] = data_1['company_type'].fillna('Pvt Ltd')
data_1['company_size'] = data_1['company_size'].fillna(method = 'ffill')
data_1['last_new_job'] = data_1['last_new_job'].fillna(method = 'bfill')
data_1['gender'] = data_1['gender'].fillna(data_1['gender'].mode()[0])
data_1['major_discipline'] = data_1['major_discipline'].fillna(data_1['major_discipline'].mode()[0])
data_1


# In[131]:


data_2 = data_1.copy()
data_2 = data_2.dropna()


# In[114]:


data_2.shape


# In[132]:


missing_2=data_2.isnull().sum()
missing_2


# In[116]:


correlation=data_2.corr()
sns.heatmap(correlation,annot=True)
plt.show()


# In[133]:


data_num=data[['city','city_development_index','training_hours']]
data_cat=data[['gender','relevent_experience','enrolled_university','education_level','major_discipline','company_type']]


# In[134]:


for i in data_num.columns:
    plt.hist(data_num[i])
    plt.title(i)
    plt.show()


# In[135]:


pd.pivot_table(data,index='target',values=['city','city_development_index','training_hours'])


# In[136]:


for i in data_cat.columns:
    sns.barplot(data_cat[i].value_counts().index,data_cat[i].value_counts()).set_title(i)
    plt.show()


# In[93]:


pd.pivot_table(data,index='target',values=data_cat,aggfunc='count')


# In[94]:


pd.pivot_table(data_1,index='target',values=data_cat,aggfunc='count')


# In[137]:


ordinal_experience = {'<1':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10,
                      '11':11, '12':12, '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20, '>20':21}


# In[138]:


data_2.experience = data_2.experience.map(ordinal_experience)


# In[139]:


ordinal_company_size = {'<10':0, '10/49':1, '50-99':2, '100-500':3, '500-999':4, '1000-4999':5, '5000-9999':6, '10000+':7}
data_2.company_size = data_2.company_size.map(ordinal_company_size)


# In[140]:


ordinal_last_new_job = {'never':0, '1':1, '2':2, '3':3, '4':4, '>4':5}
data_2.last_new_job = data_2.last_new_job.map(ordinal_last_new_job)


# In[141]:


data_2


# In[142]:


def categories(multi_columns):
    final=data_2
    i=0
    for field in multi_columns:
        
        print(field)
        data_3=pd.get_dummies(data_2[field],drop_first=True)
        data_2.drop([field],axis=1,inplace=True)
        if i == 0:
            final=data_3.copy()
        else:
            final=pd.concat([final,data_3],axis=1)
        i=i+1
    final=pd.concat([data_2,final],axis=1)
    
    return final
final_data_1=categories(data_cat)


# In[144]:


final_data_1.nunique()
missing_3 = final_data_1.isnull().sum()
missing_3
final_data_1.info()


# In[145]:


final_data=final_data_1.drop(columns=["enrollee_id"],axis=1)
X=final_data.drop(['target'],axis=1)
Y=final_data['target']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
X_train


# In[146]:


logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred= logistic.predict(X_test)
score_1=accuracy_score(y_test,y_pred)
print("Accuracy on Traing set: ",logistic.score(X_train,y_train))
print("Accuracy on Testing set: ",logistic.score(X_test,y_test))
print("accuracy_score", score_1)


# In[ ]:




