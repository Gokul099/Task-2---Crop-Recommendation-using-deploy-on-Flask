import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)

df = pd.read_csv('Crop_recommendation.csv')
df.head()

df.rename(columns = {'District_Name':'village'},inplace = True)

df.head()

"""# Basic Checks"""

df.tail()

df.info()

df.shape

df.describe()

df.describe(include = ['O'])

"""# EDA"""

num = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(10, 10))
plotnumber = 1
for nums in num:
    if plotnumber <= len(num):
        plt.subplot(3, 3, plotnumber)
        sns.distplot(df[nums].dropna(axis=0))
        plt.xlabel(nums)
        plt.ylabel('count')
        plotnumber += 1

plt.tight_layout()
plt.show()

"""# Insights
### check distribution: all features are non distribution.
"""

df.drop(df[df['village'] == 'E_Perumalpatti'].index, inplace=True)

df.village.value_counts()

df.village.replace(['Elumalai','Elumalai '] ,'Elumalai', inplace = True)
df.Crop.replace(['cotton','Cotton'],'cotton', inplace = True)

# Village
plt.figure(figsize = (15,10))
sns.countplot(x = 'village', data = df)
plt.xticks(fontsize = 15,rotation = 90)
plt.show()

"""## Insights
### Anaikaraipatti and krishanapuram is highest and Mallaparam and  t_paraipatti
"""

df.Soil_color.astype('object')

df.Soil_color.value_counts

plt.figure(figsize = (12,8))
sns.countplot(x = 'Soil_color',data = df)
plt.show()

"""## Insights
### red is more
"""

plt.figure(figsize = (12,9))
sns.countplot(x = 'Crop',data = df)
plt.xticks(rotation = 90, fontsize = 15)
plt.show()

"""## Insights
### Corn and rice is highest and jowar and onion is low
"""

#village * soil
col = ('red','black','green')
plt.figure(figsize = (15,10))
sns.countplot(x = 'village',data = df,hue = 'Soil_color', palette = col)
plt.xticks(rotation = 90)
plt.show()

"""## Insights
### Every village red soil is more and less alluvil
"""

3#village * label
plt.figure(figsize = (15,10))
sns.countplot(x = 'village',data = df,hue = 'Crop')
plt.xticks(rotation = 90)
plt.show()

"""## Insights
### Every village corn is more and some villafe produce rice only, and least count is blackeyedpeans
"""

#soil * LAbel
color = ('red','black','green')
plt.figure(figsize = (13,9))
sns.countplot(x = 'Crop',data = df, hue = 'Soil_color', palette = color)
plt.show()

"""## Insights
### Every crop growth in Red soil, paricuraly corn, and alluvali is less
"""

df.head(2)

# Nitrogen * phosphorus
plt.figure(figsize = (12,9))
sns.scatterplot(x = 'Nitrogen',y = 'Phosphorus', data = df,hue = 'Crop')
plt.show()

"""## Insights barmyarmillet
### low Nitrogen and low phosphorus more corn produced , and more nitrogen and more phosphorus  producrd cotton and  barmyarmillet
"""

#Nitrogen * Potassium
plt.figure(figsize = (12,9))
sns.scatterplot(x = 'Nitrogen',y = 'Potassium', data = df,hue = 'Crop')
plt.show()

"""## Insights
### less nitrogen and less potassium produced more corn and high is rice
"""

# label * Rainfall
plt.figure(figsize = (15,10))
sns.histplot(x = 'Rainfall', data = df,hue = 'Crop')
plt.show()

"""## Insights
### 100 to 1000 rainfall is due to produce cotton and jowar more but above 1000 rainfall rice and jowar and barnyardmillet is more
"""

#village * rainfall
plt.figure(figsize = (15,10))
sns.histplot(x = 'Rainfall', data = df,hue = 'village')
plt.show()

"""## Insights
### more rainfall in Anaikaranpattiand saranpatti]
"""

# Potassium * corn
plt.figure(figsize = (15,10))
sns.histplot(x = 'Potassium', data = df,hue = 'Crop')
plt.show()

"""## Insights
### potassium less for corn and rice more used in jowar and rice

# preprocessing
"""

df.duplicated().sum()

df.isnull().sum()

df1 = df.copy()

df1

df.loc[df['village'].isnull() == True,'village']

df.drop(903, inplace=True)

"""# Encoding"""

df.Soil_color.value_counts()

df.replace({'red':1,"black":2,"alluvial ":3},inplace = True)

df

df.Crop.astype('object')

df.drop(['village'],axis = 1,inplace = True)

df.head(2)

df.replace({'corn':0,'rice':1,'cotton':2,'barnyardmillet':3,'groundnut':4,'Sugarcane':5,'blackeyedpeas':6,'chilly':7,'Jowar':8,'onion':9},inplace = True)

"""# split

"""

x = df.drop('Crop',axis = 1)
y = df['Crop']

x.head()

y.head()

y.info()

y.value_counts()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

x_train.shape

y_test.shape

"""# model ceation"""

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(x_train,y_train)

pred = xg.predict(x_test)
a = accuracy_score(y_test,pred)
print(a)

import joblib

file = 'crop'
joblib.dump(xg,'crop')
app = joblib.load('crop')


import pickle

Pkl_Filename = "crop_model.pkl"  


with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(xg, file)


with open(Pkl_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)

