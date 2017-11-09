# coding=UTF-8
import pandas as pd
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv("../input/train.csv")


def age_map(age):
    result = 8
    if (age < 10):
        result = 0
    elif (age < 20):
        result = 1
    elif (age < 30):
        result = 2
    elif (age < 40):
        result = 3
    elif (age < 50):
        result = 4
    elif (age < 60):
        result = 5
    elif (age < 70):
        result = 6
    elif (age < 90):
        result = 7

    return result


def get_dummy_cats(df):
    return (pd.get_dummies(df, columns=['Pclass', 'Sex_Val',
                                        'AgeFill', 'FamilySize', 'Fare_cat']))


def get_title_last_name(name):
    full_name = name.str.split(', ', n=0, expand=True)
    last_name = full_name[0]
    titles = full_name[1].str.split('.', n=0, expand=True)
    titles = titles[0]
    return (titles)


def get_titles_from_names(df):
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df['Title'] = df['Title'].map(title_map)
    df['Title'] = df['Title'].fillna(0)
    # df['Title'] = get_title_last_name(df['Name'])
    return (df)


def clean_data(df, drop_passenger_id):
    # Get the unique values of Sex
    sexes = sorted(df['Sex'].unique())

    # Generate a mapping of Sex from a string to a number representation
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))

    # Transform Sex from a string to a number representation
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)
    df["Embarked"] = df["Embarked"].fillna("S")
    df = get_titles_from_names(df)
    # Get the unique values of Embarked
    embarked_locs = sorted(df['Embarked'].unique())

    # Generate a mapping of Embarked from a string to a number representation
    embarked_locs_mapping = dict(zip(embarked_locs,
                                     range(0, len(embarked_locs) + 1)))

    # Transform Embarked from a string to dummy variables
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked_Val')], axis=1)

    # Fill in missing values of Embarked
    # Since the vast majority of passengers embarked in 'S': 3,
    # we assign the missing values in Embarked to 'S':
    # if len(df[df['Embarked'].isnull()] > 0):
    #     df.replace({'Embarked_Val':
    #                     {embarked_locs_mapping['nan']: embarked_locs_mapping['S']
    #                      }
    #                 },
    #                inplace=True)

    # Fill in missing values of Fare with the average Fare
    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({None: avg_fare}, inplace=True)

    # To keep Age in tact, make a copy of it called AgeFill
    # that we will use to fill in the missing ages:
    df['AgeFill'] = df['Age']

    # Determine the Age typical for each passenger class by Sex_Val.
    # We'll use the median instead of the mean because the Age
    # histogram seems to be right skewed.
    df['AgeFill'] = df['AgeFill'] \
        .groupby([df['Sex_Val'], df['Pclass']]) \
        .apply(lambda x: x.fillna(x.median()))

    # Define a new feature FamilySize that is the sum of
    # Parch (number of parents or children on board) and
    # SibSp (number of siblings or spouses):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    # df['NameLength'] = df['Name'].apply(lambda x:len(x))
    df['AgeFill'] = df['AgeFill'].apply(lambda x: age_map(x))

    # Drop the columns we won't use:
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    df['Fare_cat'] = 0
    df.loc[df['Fare'] <= 7.91, 'Fare_cat'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare_cat'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare_cat'] = 2
    df.loc[(df['Fare'] > 31) & (df['Fare'] <= 513), 'Fare_cat'] = 3
    # Drop the Age column since we will be using the AgeFill column instead.
    # Drop the SibSp and Parch columns since we will be using FamilySize.
    # Drop the PassengerId column since it won't be used as a feature.
    df = df.drop(['Age', 'SibSp', 'Parch', 'Fare'], axis=1)

    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)

    return get_dummy_cats(df)


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_features='log2', min_samples_leaf=1,min_samples_split=2,criterion='entropy', n_estimators=250, max_depth=6, verbose=0)
clf2 = RandomForestClassifier()
# train_target=df_train['Survived'].values

df_train = clean_data(df_train, drop_passenger_id=False)
# print df_train.head(10)
# exit()
train_data = df_train.values
# print df_train.head(10)
# Training data features, skip the first column 'Survived'
train_features = train_data[:, 2:]  # Fit the model to our training data
# print df_train.head(10)
# 'Survived' column values
train_target = train_data[:, 1]
# print train_features.shape
clf = clf.fit(train_features, train_target)
score = clf.score(train_features, train_target)
print "Mean accuracy of Random Forest: {0}".format(score)

# predicting
df_test = pd.read_csv('../input/test.csv')
# Data wrangle the test set and convert it to a numpy array
df_test = clean_data(df_test, drop_passenger_id=False)
# print df_test.head(10)
test_data = df_test.values
# Get the test data features, skipping the first column 'PassengerId'
test_x = test_data[:, 1:]
# print test_x.shape
test_x1 = test_x
# Predict the Survival values for the test data
test_y = clf.predict(test_x).astype(int)

# 预测概率
# test_y = clf.predict_log_proba(test_x)
# test_y = clf.predict_proba(test_x)
# df_test['Survived'] = test_y
# df_test[['PassengerId', 'Survived']] \
#     .to_csv('../input/submit2.csv', index=False)

from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# 支持向量机
# clf = svm.SVC()
# 逻辑回归
# clf = LogisticRegression()
# clf = GradientBoostingClassifier(random_state=100)
# Split 80-20 train vs test data
train_x, test_x, train_y, test_y = train_test_split(train_features,
                                                    train_target,
                                                    test_size=0.20,
                                                    random_state=0)

cross_validation = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)

param_grid = {
    'n_estimators': [200, 250, 300, 400],
    'min_samples_split': [2, 3, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': [1,2, 3, 4],
    'max_depth': [6, 7, 8, 9]
}

# gsearch = GridSearchCV(estimator=clf2, param_grid=param_grid, cv=cross_validation)
# gsearch.fit(train_features, train_target)
# print gsearch.grid_scores_
# print gsearch.best_params_
# print gsearch.best_score_
# exit()
#
# print (train_features.shape, train_target.shape)
# print (train_x.shape, train_y.shape)
# print (test_x.shape, test_y.shape)
# clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)

test_y2 = clf.predict(test_x1).astype(int)

# 预测概率
# test_y = clf.predict_log_proba(test_x)
# test_y = clf.predict_proba(test_x)
df_test['Survived'] = test_y2
df_test[['PassengerId', 'Survived']] \
    .to_csv('../input/random_forest_gridsearch_submit2.csv', index=False)

# 衡量指标
from sklearn.metrics import accuracy_score

print ("Accuracy = %.4f" % (accuracy_score(test_y, predict_y)))
model_score = clf.score(test_x, test_y)
print ("Model Score %.4f \n" % (model_score))
print ("AUC Score %.4f \n" % (metrics.roc_auc_score(test_y, predict_y)))

confusion_matrix = metrics.confusion_matrix(test_y, predict_y)
print ("Confusion Matrix ", confusion_matrix)

print ("          Predicted")
print ("         |  0  |  1  |")
print ("         |-----|-----|")
print ("       0 | %3d | %3d |" % (confusion_matrix[0, 0],
                                   confusion_matrix[0, 1]))
print ("Actual   |-----|-----|")
print ("       1 | %3d | %3d |" % (confusion_matrix[1, 0],
                                   confusion_matrix[1, 1]))
print ("         |-----|-----|")
