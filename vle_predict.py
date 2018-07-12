import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

studentInfo = pd.read_csv("studentInfo.csv",  delimiter=',', na_values='?', index_col=False)
studentVle = pd.read_csv("studentVle.csv",  delimiter=',', na_values='?', index_col=False)
vle = pd.read_csv("vle.csv",  delimiter=',', index_col=False, na_values='?')

data_demog = studentInfo[['id_student','code_module', 'gender', 'region',
                          'highest_education', 'code_presentation','imd_band', 'age_band', 'num_of_prev_attempts',
                          'studied_credits', 'disability', 'final_result']]

# pick the vle data for early days in semester. Merge with activity type.
data_vle_early = studentVle.where(studentVle['date'] < 50)
data_activity = data_vle_early.merge(vle[['id_site', 'activity_type']], on='id_site')

data_new = pd.pivot_table(data_activity, values='sum_click', index='id_student', columns='activity_type',
                          aggfunc=np.mean).reset_index()
data = data_demog.merge(data_new, on='id_student')
data_size, feature_size = data.shape

# pre-processing
# do not use the id in the predictive model because it is not a feature! set the target
target = data['final_result'].copy()
data = data.drop(['id_student', 'final_result'], 1)

# remove columns with high number of missing value
data = data.dropna(axis=1, thresh=int(3*data_size/4))
feature_names = data.columns


# 'code_presentation' is to be changed to the semester the course is presented. This function is to seperate that letter
def getletter(variable, letternumber):
    return str(variable)[letternumber - 1]

for _ in range(len(data['code_presentation'])):
    data['code_presentation'][_] = getletter(data['code_presentation'][_], 5)


# Impute the missing values, median method works for both cat and con data type,
# be careful about axis! I will be using this after I transformed my data to numpy array.
imputer = Imputer(strategy='median', axis=0)

# Encode the categorical data
encode = LabelEncoder()
data[['code_module', 'gender', 'region', 'highest_education', 'code_presentation', 'imd_band', 'age_band',
      'disability']] \
    = data[['code_module', 'gender', 'region', 'highest_education', 'code_presentation', 'imd_band', 'age_band',
            'disability']].astype(str).apply(encode.fit_transform)

# the following labeling changes the multi-class problem into two class to recognize failure or withdrawal
for i in range(len(target)):
    if target[i] == 'Fail':
        target[i] = 1
    else:
        target[i] = 0

# change to numpy array to work with sklearn
target = np.asarray(target).reshape((data_size,)).astype(int)
data = np.asarray(data)


# pipeline for training and testing using different classifiers to see which one works better
c = [
    GaussianNB(),
    # XGBClassifier(n_estimators=100),
    # KNeighborsClassifier(),
    # svm.SVC(kernel='rbf', C=0.1, class_weight='balanced'),
    ExtraTreesClassifier(n_estimators=100),
    RandomForestClassifier(n_estimators=100, max_depth=4)
    ]

for clf in c:
    # print(clf)
    pipeline = Pipeline(steps=[('Impute', imputer),
                               ('clf', clf)])

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)
    pipeline.fit(data_train, target_train)
    accuracy_train = pipeline.score(data_train, target_train)
    accuracy = pipeline.score(data_test, target_test)
    print(accuracy, accuracy_train)

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        feature_ranking = list(zip(importances, feature_names))
        ranked = sorted(feature_ranking, reverse=True)
        print(ranked)

# importance of features including VLE

