# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:34:40 2019

@author: hp
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import pickle

data = pd.read_csv('train_.csv')
test_data = pd.read_csv('test_.csv')
#data = train_data.append(test_data, ignore_index=True)
y = data['is_promoted']

data = data.drop(['is_promoted'],axis = 1)

dept_counts = data['department'].value_counts()
region_count = data['region'].value_counts()
region_data = data['region'].str.replace("[a-zA-Z_]","")
data['region']= data['region'].str.replace("[a-zA-Z_]","")
region = data['region'].astype(int)
region = region.astype(int)

data = pd.get_dummies(data, columns=['gender']) 
data = data.drop(['gender_f'],axis = 1)

data = pd.get_dummies(data, columns=['education']) 
data = data.drop(['education_Below Secondary'],axis = 1)

data = pd.get_dummies(data, columns=['recruitment_channel']) 
data = data.drop(['recruitment_channel_referred'],axis = 1)

from sklearn.preprocessing import LabelBinarizer
lb_style = LabelBinarizer()
lb = lb_style.fit_transform(data["department"])
data['previous_year_rating'] = data['previous_year_rating'].fillna(data['previous_year_rating'].median())

data = data.drop(['department'],axis = 1)
d1 =data.insert(1,'Region',region)
data = data.drop(['region'],axis = 1)
d = data

count_ofall_nan = data.isna().sum()
X = data.iloc[:,0:14].values
X = np.hstack((X,lb))
count_ = np.isnan(np.sum(lb))
data = data.astype(np.int64)
#train = X[:54808,:]
#test = X[54808:,:]
##count_ = np.isnan(np.sum(train))
##pos = np.argwhere(np.isnan(train))
#
#from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imp.fit(train)
#train= imp.transform(train)
#
#train = np.nan_to_num(train)
#train = train.astype(np.int64)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X, y)
pickle.dump(nb, open('gNB.sav','wb'))
#
#train = np.isnan(train)
#pos = np.argwhere(np.isnan(train))
#
#random forest classifier
from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier(n_estimators=100)
random.fit(X,y)
pickle.dump(random, open('random_forest.sav','wb'))

from sklearn.naive_bayes import MultinomialNB
classifier_multi = MultinomialNB()
classifier_multi.fit(X, y)
pickle.dump(classifier_multi, open('classifier_multi_NB.sav','wb'))



#//////////////////////////////////////////////////////////////////////////

#data = data.drop(['is_promoted'],axis = 1)

#dept_counts = test_data['department'].value_counts()
#region_count = test_data['region'].value_counts()
region_data = test_data['region'].str.replace("[a-zA-Z_]","")
test_data['region']= test_data['region'].str.replace("[a-zA-Z_]","")
region = test_data['region'].astype(int)
region = region.astype(int)

y_true = test_data['is_promoted']
test_data = test_data.drop(['is_promoted'],axis = 1)

test_data = pd.get_dummies(test_data, columns=['gender']) 
test_data = test_data.drop(['gender_f'],axis = 1)

test_data = pd.get_dummies(test_data, columns=['education']) 
test_data = test_data.drop(['education_Below Secondary'],axis = 1)

test_data = pd.get_dummies(test_data, columns=['recruitment_channel']) 
test_data = test_data.drop(['recruitment_channel_referred'],axis = 1)

from sklearn.preprocessing import LabelBinarizer
lb_style_t = LabelBinarizer()
lb_test = lb_style_t.fit_transform(test_data["department"])
test_data['previous_year_rating'] = test_data['previous_year_rating'].fillna(test_data['previous_year_rating'].median())

test_data = test_data.drop(['department'],axis = 1)
d2 =test_data.insert(1,'Region',region)
test_data = test_data.drop(['region'],axis = 1)

X_test = test_data.iloc[:,0:14].values
X_test = np.hstack((X_test,lb_test))

x_test_df = pd.DataFrame(X_test)
x_test_df.to_csv('test_preprocessed.csv')

ypred = nb.predict(X_test)
ypred_randomF = random.predict(X_test)
ypred_multi = classifier_multi.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_gnb = confusion_matrix(y_true, ypred)
cm_rf = confusion_matrix(y_true,ypred_randomF)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true,ypred)
acc_rf = accuracy_score(y_true,ypred_randomF)


#from mrcnn.config import Config
#from mrcnn.model import MaskRCNN
#from mrcnn.utils import Dataset
#from mrcnn.utils import compute_ap
#from mrcnn.model import load_image_gt
#from mrcnn.model import mold_image
#
#
#
#import cv2
 