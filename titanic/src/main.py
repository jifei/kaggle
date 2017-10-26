# coding=UTF-8
import numpy as np
import pandas as pd
import pylab as plt
# Set the global default size of matplotlib figures
plt.rc('figure', figsize=(10, 5))

# Size of matplotlib figures that contain subplots
fizsize_with_subplots = (10, 10)

# Size of matplotlib histogram bins
bin_size = 10

df_train = pd.read_csv("../input/train.csv")
# 特征工程
print df_train.describe()
print df_train.head(10)
# print df_train.describe()
print df_train["Sex"].unique()
print df_train["Embarked"].unique()
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())
df_train["Embarked"] = df_train["Embarked"].fillna("S")
# df_train.loc[df_train["Embarked"] == "S", "Embarked"] = 0
# df_train.loc[df_train["Embarked"] == "C", "Embarked"] = 1
# df_train.loc[df_train["Embarked"] == "Q", "Embarked"] = 2
sexes = sorted(df_train['Sex'].unique())
print sexes
# df_train.loc[df_train["Sex"] == "male", "Sex"] = 0
# df_train.loc[df_train["Sex"] == "female", "Sex"] = 1


# Set up a grid of plots
fig = plt.figure(figsize=fizsize_with_subplots)
fig_dims = (3, 2)

# Plot death and survival counts
plt.subplot2grid(fig_dims, (0, 0))
df_train['Survived'].value_counts().plot(kind='bar',
                                         title='Death and Survival Counts')

# Plot Pclass counts
plt.subplot2grid(fig_dims, (0, 1))
df_train['Pclass'].value_counts().plot(kind='bar',
                                       title='Passenger Class Counts')

# Plot Sex counts
plt.subplot2grid(fig_dims, (1, 0))
df_train['Sex'].value_counts().plot(kind='bar',
                                    title='Gender Counts')
plt.xticks(rotation=0)

# Plot Embarked counts
plt.subplot2grid(fig_dims, (1, 1))
df_train['Embarked'].value_counts().plot(kind='bar',
                                         title='Ports of Embarkation Counts')

# Plot the Age histogram
plt.subplot2grid(fig_dims, (2, 0))
df_train['Age'].hist()
plt.title('Age Histogram')
# plt.show()




pclass_xt = pd.crosstab(df_train['Pclass'], df_train['Survived'])
# Normalize the cross tab to sum to 1:
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)

pclass_xt_pct.plot(kind='bar',
                   stacked=True,
                   title='Survival Rate by Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
# plt.show()



genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
print genders_mapping
df_train['Sex_Val'] = df_train['Sex'].map(genders_mapping).astype(int)
print  pd.crosstab(df_train['Sex_Val'], df_train['Survived'])
print df_train.head(10)
sex_val_xt = pd.crosstab(df_train['Sex_Val'], df_train['Survived'])
sex_val_xt_pct = sex_val_xt.div(sex_val_xt.sum(1).astype(float), axis=0)
sex_val_xt_pct.plot(kind='bar', stacked=True, title='Survival Rate by Gender')
# plt.show()

#乘客班次
passenger_classes = sorted(df_train['Pclass'].unique())
print passenger_classes
for p_class in passenger_classes:
    print 'M: ', p_class, len(df_train[(df_train['Sex'] == 'male') &
                             (df_train['Pclass'] == p_class)])
    print 'F: ', p_class, len(df_train[(df_train['Sex'] == 'female') &
                             (df_train['Pclass'] == p_class)])
print pd.crosstab(df_train['Pclass'], df_train['Survived'])


# Plot survival rate by Sex
females_df = df_train[df_train['Sex'] == 'female']
females_xt = pd.crosstab(females_df['Pclass'], df_train['Survived'])
females_xt_pct = females_xt.div(females_xt.sum(1).astype(float), axis=0)
females_xt_pct.plot(kind='bar',
                    stacked=True,
                    title='Female Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

# Plot survival rate by Pclass
males_df = df_train[df_train['Sex'] == 'male']
males_xt = pd.crosstab(males_df['Pclass'], df_train['Survived'])
males_xt_pct = males_xt.div(males_xt.sum(1).astype(float), axis=0)
males_xt_pct.plot(kind='bar',
                  stacked=True,
                  title='Male Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
# plt.show()

embarked_locs = sorted(df_train['Embarked'].unique())

embarked_locs_mapping = dict(zip(embarked_locs,
                                 range(0, len(embarked_locs) + 1)))
print embarked_locs_mapping
df_train['Embarked_Val'] = df_train['Embarked'] \
                               .map(embarked_locs_mapping) \
                               .astype(int)
df_train.head()



df_train['Embarked_Val'].hist(bins=len(embarked_locs), range=(0, 3))
plt.title('Port of Embarkation Histogram')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')
# plt.show()
df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked_Val'], prefix='Embarked_Val')], axis=1)

df_train[df_train['Age'].isnull()][['Sex', 'Pclass', 'Age']].head()
df_train['AgeFill'] = df_train['Age']

# Populate AgeFill
df_train['AgeFill'] = df_train['AgeFill'] \
                        .groupby([df_train['Sex_Val'], df_train['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))


df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']

def clean_data(df, drop_passenger_id):
    # Get the unique values of Sex
    sexes = sorted(df['Sex'].unique())

    # Generate a mapping of Sex from a string to a number representation
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))

    # Transform Sex from a string to a number representation
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)

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
    if len(df[df['Embarked'].isnull()] > 0):
        df.replace({'Embarked_Val':
                        {embarked_locs_mapping[nan]: embarked_locs_mapping['S']
                         }
                    },
                   inplace=True)

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

    # Drop the columns we won't use:
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # Drop the Age column since we will be using the AgeFill column instead.
    # Drop the SibSp and Parch columns since we will be using FamilySize.
    # Drop the PassengerId column since it won't be used as a feature.
    df = df.drop(['Age', 'SibSp', 'Parch'], axis=1)

    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)

    return df

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)



df_train = df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],
                         axis=1)


df_train = df_train.drop(['Age', 'SibSp', 'Parch', 'PassengerId', 'Embarked_Val'], axis=1)
print df_train.head(10)


train_data = df_train.values
# Training data features, skip the first column 'Survived'
train_features = train_data[:, 1:]

# 'Survived' column values
train_target = train_data[:, 0]
print train_features.shape



# Fit the model to our training data
clf = clf.fit(train_features, train_target)
score = clf.score(train_features, train_target)
print "Mean accuracy of Random Forest: {0}".format(score)


#predicting
df_test = pd.read_csv('../input/test.csv')
df_test.head()
# Data wrangle the test set and convert it to a numpy array
df_test = clean_data(df_test, drop_passenger_id=False)
print df_test.head(10)
test_data = df_test.values
# Get the test data features, skipping the first column 'PassengerId'
test_x = test_data[:, 1:]
print test_x.shape
# Predict the Survival values for the test data
test_y = clf.predict(test_x)
df_test['Survived'] = test_y
df_test[['PassengerId', 'Survived']] \
    .to_csv('../input/submit.csv', index=False)


