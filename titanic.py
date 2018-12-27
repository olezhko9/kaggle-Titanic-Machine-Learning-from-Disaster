import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_test_data = [train, test]  # combine dataset
# анализируем данные

# шанс на выживание в зависимости от пола
# ax = sns.barplot(x='Sex', y='Survived', data=train)
# plt.show()
#
# # шанс на выживание в зависимости от пола и возраста
# fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
# bins = 18
# axid = 0
# for sex in ['female', 'male']:
#     for surv in [1, 0]:
#         ax = sns.distplot(train[(train['Sex'] == sex) & (train['Survived'] == surv)].Age.dropna(),
#                   bins=bins, label=str(surv), ax=axes[axid], kde=False)
#     axid += 1
#     ax.legend()
#     ax.set_title(sex)
# plt.show()
#
# # шанс на выживание в зависимости от класса билета
# ax = sns.barplot(x='Pclass', y='Survived', data=train)
# plt.show()


# for dataset in train_test_data:
#     dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
#                  "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
#                  "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}


# Пропущенные данные
total = train.isnull().sum().sort_values(ascending=False)
print(total.head(5))

#
for dataset in train_test_data:
    # dataset['Title'] = dataset["Title"].map(title_mapping)
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map(titles)

# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby('Title')['Age'].transform("median"), inplace=True)

# all_data = pd.concat([train, test])
# age_sex_median = all_data.groupby("Title")["Age"].median()
# all_data["Age"] = all_data.apply(lambda raw: age_sex_median[raw["Title"]] if pd.isnull(raw["Age"]) else raw["Age"], axis=1)
# all_data['Age'] = all_data['Age'].astype(int)
# all_data['AgeBand'] = pd.cut(all_data['Age'], 5)
# print(all_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
# all_data = all_data.drop('AgeBand', axis=1)
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3,
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[dataset['Fare'] >= 100, 'Fare'] = 3

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

# train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
# test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

# family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    # dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
    # семья
    dataset['Family'] = dataset["Parch"] + dataset["SibSp"] + 1
    dataset['Family'].loc[dataset['Family'] > 1] = 1
    dataset['Family'].loc[dataset['Family'] == 1] = 0


features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
clf = [
    KNeighborsClassifier(n_neighbors=13),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=13),
    GaussianNB(),
    SVC(),
    ExtraTreeClassifier(),
    GradientBoostingClassifier(n_estimators=10, learning_rate=1, max_features=3, max_depth=3, random_state=10),
    AdaBoostClassifier(),
    ExtraTreesClassifier()
]


def model_fit():
    scoring = 'accuracy'
    for i in range(len(clf)):
        score = cross_val_score(clf[i], train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
        print("Score of Model", i, ":", round(np.mean(score) * 100, 2))


#     round(np.mean(score)*100,2)
#     print("Score of :\n",score)
model_fit()

clf1 = RandomForestClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=16, n_estimators=400)
clf1.fit(train_data, target)
test_data = test.drop('PassengerId', axis=1)
prediction = clf1.predict(test_data)

# boost
# param_grid = {"criterion": ["gini", "entropy"],
#               "min_samples_leaf": [1, 5, 10, 25, 50, 70],
#               "min_samples_split": [2, 4, 10, 12, 16, 18, 25, 35],
#               "n_estimators": [100, 400, 700, 1000, 1500]
#               }
#
# rf = RandomForestClassifier(max_features='auto', random_state=1, n_jobs=12)
# clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=12)
# clf.fit(train_data, target)
# print(clf.best_params_)

test_data['Survived'] = prediction
submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": test_data['Survived']
})
submission.to_csv("Submission.csv", index=False)
