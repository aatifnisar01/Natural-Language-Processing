# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 22:17:45 2022

@author: Aatif
"""

import pandas as pd
df=pd.read_csv('Data.csv',encoding="ISO_8859_1")
train=df[df['Date']<'20150101']
test=df[df['Date']>'20141231']

data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]",' ',regex=True,inplace=True)

list1=[i for i in range(25)]
new_index=[str(i) for i in list1]
data.columns=new_index


for index in new_index:
    data[index]=data[index].str.lower()
    
headlines=[]
for row in range(0,len(data.index)):
    headlines.append(''.join(str(x) for x in data.iloc[row,0:25]))
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(2,2))
from sklearn.ensemble import RandomForestClassifier

traindataset=cv.fit_transform(headlines)
    
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])

test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(''.join(str(x) for x in test.iloc[row,2:27]))
    
test_dataset=cv.transform(test_transform)
predictions=randomclassifier.predict(test_dataset)
    
from sklearn.metrics import accuracy_score
score=accuracy_score(test['Label'] , predictions)