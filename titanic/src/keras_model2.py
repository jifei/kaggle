# from __future__ import print_function
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout

raw_train = pd.read_csv('../input/train.csv', index_col=0)
raw_train['is_test'] = 0
raw_test = pd.read_csv('../input/test.csv', index_col=0)
raw_test['is_test'] = 1
all_data = pd.concat((raw_train, raw_test), axis=0)


def get_title_last_name(name):
    full_name = name.str.split(', ', n=0, expand=True)
    last_name = full_name[0]
    titles = full_name[1].str.split('.', n=0, expand=True)
    titles = titles[0]
    return (titles)


def get_titles_from_names(df):
    df['Title'] = get_title_last_name(df['Name'])
    df = df.drop(['Name'], axis=1)
    return (df)


def get_dummy_cats(df):
    return (pd.get_dummies(df, columns=['Title', 'Pclass', 'Sex', 'Embarked',
                                        'Cabin', 'Cabin_letter']))


def get_cabin_letter(df):
    df['Cabin'].fillna('Z', inplace=True)
    df['Cabin_letter'] = df['Cabin'].str[0]
    return (df)


def process_data(df):
    # preprocess titles, cabin, embarked
    df = get_titles_from_names(df)
    df['Embarked'].fillna('S', inplace=True)
    df = get_cabin_letter(df)

    # drop remaining features
    df = df.drop(['Ticket', 'Fare'], axis=1)

    # create dummies for categorial features
    df = get_dummy_cats(df)

    return (df)

def clean_data(df, drop_passenger_id):
    # Get the unique values of Sex
    sexes = sorted(df['Sex'].unique())

    # Generate a mapping of Sex from a string to a number representation
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))

    # Transform Sex from a string to a number representation
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)

    # Fill in missing values of Fare with the average Fare
    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({None: avg_fare}, inplace=True)

    # To keep Age in tact, make a copy of it called AgeFill
    # that we will use to fill in the missing ages:

    # Determine the Age typical for each passenger class by Sex_Val.
    # We'll use the median instead of the mean because the Age
    # histogram seems to be right skewed.
    df['Age'] = df['Age'] \
        .groupby([df['Sex_Val'], df['Pclass']]) \
        .apply(lambda x: x.fillna(x.median()))

    # Define a new feature FamilySize that is the sum of
    # Parch (number of parents or children on board) and
    # SibSp (number of siblings or spouses):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    # df['NameLength'] = df['Name'].apply(lambda x:len(x))

    # Drop the columns we won't use:
    df['Sex']=df['Sex_Val']
    df = df.drop(['Sex_Val'], axis=1)

    # Drop the Age column since we will be using the AgeFill column instead.
    # Drop the SibSp and Parch columns since we will be using FamilySize.
    # Drop the PassengerId column since it won't be used as a feature.
    # df = df.drop(['Age', 'SibSp', 'Parch'], axis=1)

    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)

    return df

# print all_data.head(10)
# print clean_data(all_data,drop_passenger_id=False).head(10)
# exit()
proc_data = process_data(clean_data(all_data,drop_passenger_id=False))
proc_train = proc_data[proc_data['is_test'] == 0]
proc_test = proc_data[proc_data['is_test'] == 1]
# print proc_test.loc[proc_test['Age'].isnull()]['Age'].head(10)
# print proc_test['Age'].loc[proc_test['Age'].isnull()].shape
# exit(0)


# Build Network to predict missing ages
for_age_train = proc_data.drop(['Survived', 'is_test'], axis=1).dropna(axis=0)
X_train_age = for_age_train.drop('Age', axis=1)
y_train_age = for_age_train['Age']

# # create model
# tmodel = Sequential()
# tmodel.add(Dense(input_dim=X_train_age.shape[1], units=128,
#                  kernel_initializer='normal', bias_initializer='zeros'))
# tmodel.add(Activation('relu'))
#
# for i in range(0, 8):
#     tmodel.add(Dense(units=64, kernel_initializer='normal',
#                      bias_initializer='zeros'))
#     tmodel.add(Activation('relu'))
#     tmodel.add(Dropout(.2))
#
# tmodel.add(Dense(units=1))
# tmodel.add(Activation('linear'))
#
# tmodel.compile(loss='mean_squared_error', optimizer='rmsprop')
#
# tmodel.fit(X_train_age.values, y_train_age.values, epochs=600, verbose=2)

train_data = proc_train


to_pred = train_data.loc[train_data['Age'].isnull()].drop(
          ['Age', 'Survived', 'is_test'], axis=1)
# p = tmodel.predict(to_pred.values)
# print train_data['Age'].dtype.name
# print train_data.loc[train_data['Age'].isnull()].head(10)
# print p

# train_data.loc[train_data['Age'].isnull(),'Age'] = p
#
#
#
test_data = proc_test
# to_pred = test_data.loc[test_data['Age'].isnull()].drop(
#           ['Age', 'Survived', 'is_test'], axis=1)
# p = tmodel.predict(to_pred.values)
# test_data.loc[test_data['Age'].isnull(),'Age'] = p
# # print train_data.loc[train_data['Age'].isnull()]
# # print train_data['Age']

y = pd.get_dummies(train_data['Survived'])
X = train_data.drop(['Survived', 'is_test'], axis=1)


# create model
model = Sequential()
model.add(Dense(input_dim=X.shape[1], units=128,
                 kernel_initializer='normal', bias_initializer='zeros'))
model.add(Activation('relu'))

for i in range(0, 15):
    model.add(Dense(units=128, kernel_initializer='normal',
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(.2))

model.add(Dense(units=2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X.values, y.values, epochs=600, verbose=2)


p_survived = model.predict_classes(test_data.drop(['Survived', 'is_test'], axis=1).values)
submission = pd.DataFrame()
submission['PassengerId'] = test_data.index
submission['Survived'] = p_survived
submission.to_csv('../input/titanic_keras_cs2.csv', index=False)
