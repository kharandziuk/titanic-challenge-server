import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('data/train.csv')


dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
dataset['Title'] = dataset['Title'].map(title_mapping)
dataset['Title'] = dataset['Title'].fillna(0)
dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
dataset['Embarked'] = dataset['Embarked'].fillna('S')
dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

age_avg = dataset['Age'].mean()
age_std = dataset['Age'].std()
age_null_count = dataset['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
dataset['Age'] = dataset['Age'].astype(int)
dataset['AgeBand'] = pd.cut(dataset['Age'], 5)

dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)
dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
dataset['Fare'] = dataset['Fare'].astype(int)

dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

dataset['IsAlone'] = 0
dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']

lookup_dataset = dataset.drop([
    'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize',
    'PassengerId', 'AgeBand', 'FareBand', 'Survived'
], axis=1)

dataset = dataset.drop(features_drop, axis=1)
dataset = dataset.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)

# remove all the titles from name
lookup_dataset['Name'] = lookup_dataset.Name.str.replace(' ([A-Za-z]+)\.', '')

lookup_list = lookup_dataset.values.tolist()
clf = RandomForestClassifier(n_estimators=100)
X_train = dataset.drop('Survived', axis=1)
y_train = dataset['Survived']
clf.fit(X_train, y_train)

