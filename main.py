import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

# Load the dataset
fake_data = pd.read_csv('Datasets\Fake.csv')
real_data = pd.read_csv('Datasets\True.csv')

fake_data['class'] = 0
real_data['class'] = 1

fake_data_manual_testing = fake_data.tail(10)
for i in range(23480, 23470, -1):
    fake_data.drop([i], axis=0, inplace=True)

real_data_manual_testing = real_data.tail(10)
for i in range(21416, 21406, -1):
    real_data.drop([i], axis=0, inplace=True)

fake_data_manual_testing.loc[:, 'class'] = 0
real_data_manual_testing.loc[:, 'class'] = 1

merge_data = pd.concat([fake_data, real_data], axis=0)

data = merge_data.drop(['title', 'subject', 'date'], axis=1)

data = data.sample(frac=1)
data.reset_index(drop=True, inplace=True)

def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub("\n", '', text)
    text = re.sub('\w*\d\w*', '', text)

    return text

data['text'] = data['text'].apply(word_drop)

x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)

# print(DT.score(xv_test, y_test))

from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)

pred_gb = GB.predict(xv_test)
# print(GB.score(xv_test, y_test))  # Use xv_test instead of xv_train

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)

pred_rf = RF.predict(xv_test)
RF.score(xv_test, y_test)
# print(classification_report(y_test, pred_rf))

def output_lable(n):
    if n == 0:
        return False
    elif n == 1:
        return True

def manual_testing(news):
    new_def_test = pd.DataFrame({'text': [news]})
    new_def_test["text"] = new_def_test['text'].apply(word_drop)
    new_x_test = new_def_test['text']
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    lr_pred = output_lable(pred_LR[0])
    dt_pred = output_lable(pred_DT[0])
    gb_pred = output_lable(pred_GB[0])
    rf_pred = output_lable(pred_RF[0])
    
    return( lr_pred, dt_pred, gb_pred, rf_pred )

