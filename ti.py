import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix

# Data_Analysis_Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df_train = read_csv('D:\ML-Program\Titanic/train.csv')
#shape,description ,scatter matrix plot for the training data
print(' Shape = ', df_train.shape)

print('\t',df_train.head())

#Descriptive Statistics for a Dataset.
description = df_train.describe()
print(description)


#Create and Display a Scatter Plot Matrix.
scatter_matrix(df_train)
plt.show()
df_test = read_csv('D:\ML-Program\Titanic/test.csv')
#draw a bar plot of survival by sex
sns.barplot(x='Sex', y='Survived', data=df_train)
plt.show()
#print percentages of females vs. males that survive
print('Percentage of females_survived:', df_train['Survived'][df_train['Sex'] == 'female'].value_counts(normalize = True)[1]*100)

print('Percentage of males_survived:', df_train['Survived'][df_train['Sex'] == 'male'].value_counts(normalize = True)[1]*100)
#draw a bar plot of survival by Pclass
sns.barplot(x='Pclass', y='Survived', data=df_train)
plt.show()

#print percentage of people by Pclass that survived
print('Percentage of Pclass = 1_survived:', df_train['Survived'][df_train['Pclass'] == 1].value_counts(normalize = True)[1]*100)

print('Percentage of Pclass = 2_survived:', df_train['Survived'][df_train['Pclass'] == 2].value_counts(normalize = True)[1]*100)

print('Percentage of Pclass = 3_survived:', df_train['Survived'][df_train['Pclass'] == 3].value_counts(normalize = True)[1]*100)
#draw a bar plot for SibSp vs. survival
sns.barplot(x='SibSp', y='Survived', data=df_train)
plt.show()
# printing individual percent values for all of these.
print('Percentage of SibSp = 0_survived:', df_train['Survived'][df_train['SibSp'] == 0].value_counts(normalize = True)[1]*100)

print('Percentage of SibSp = 1_survived:', df_train['Survived'][df_train['SibSp'] == 1].value_counts(normalize = True)[1]*100)

print('Percentage of SibSp = 2_survived:', df_train['Survived'][df_train['SibSp'] == 2].value_counts(normalize = True)[1]*100)

#draw a bar plot for Parch vs.survival
sns.barplot(x='Parch', y='Survived', data=df_train)
plt.show()
df_train['Age'] = df_train['Age'].fillna(-0.5)
df_test['Age'] = df_test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
df_train['AgeGroup'] = pd.cut(df_train['Age'], bins, labels = labels)
df_test['AgeGroup'] = pd.cut(df_test['Age'], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x='AgeGroup', y='Survived', data=df_train)
plt.show()
df_train['CabinBool'] = (df_train['Cabin'].notnull().astype('int'))
df_test['CabinBool'] = (df_test['Cabin'].notnull().astype('int'))

#calculate percentages of CabinBool vs. survived
print('Percentage of CabinBool = 1_survived:', df_train['Survived'][df_train['CabinBool'] == 1].value_counts(normalize = True)[1]*100)

print('Percentage of CabinBool = 0_survived:', df_train['Survived'][df_train['CabinBool'] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sns.barplot(x='CabinBool', y='Survived', data=df_train)
plt.show()
sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=df_train,
              palette={'male': 'blue', 'female': 'pink'},
              markers=['*', 'o'], linestyles=['-', '--']);
plt.show()
print(df_test.describe())
# By dropping the Cabin feature since not a lot more useful information can be extracted from it.
df_train = df_train.drop(['Cabin'], axis = 1)
df_test = df_test.drop(['Cabin'], axis = 1)
#Drop the Ticket feature since it's unlikely to yield any useful information
df_train = df_train.drop(['Ticket'], axis = 1)
df_test = df_test.drop(['Ticket'], axis = 1)

# Fill  the missing values in the Embarked feature
print('Number of people embarking in Southampton (S):')
southampton = df_train[df_train['Embarked'] == 'S'].shape[0]
print(southampton)

print('Number of people embarking in Cherbourg (C):')
cherbourg = df_train[df_train['Embarked'] == 'C'].shape[0]
print(cherbourg)

print('Number of people embarking in Queenstown (Q):')
queenstown = df_train[df_train['Embarked'] == 'Q'].shape[0]
print(queenstown)
#replacing the missing values in the Embarked feature with S
df_train = df_train.fillna({'Embarked': 'S'})

##create a combined group of both datasets
combine = [df_train, df_test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(df_train['Title'], df_train['Sex']))
#replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(df_train.head())
# fill missing age with mode age group for each title
mr_age = df_train[df_train['Title'] == 1]['AgeGroup'].mode() #Young Adult
miss_age = df_train[df_train['Title'] == 2]['AgeGroup'].mode() #Student
mrs_age = df_train[df_train['Title'] == 3]['AgeGroup'].mode() #Adult
master_age = df_train[df_train['Title'] == 4]['AgeGroup'].mode() #Baby
royal_age = df_train[df_train['Title'] == 5]['AgeGroup'].mode() #Adult
rare_age = df_train[df_train['Title'] == 6]['AgeGroup'].mode() #Adult

age_title_mapping = {1: 'Young Adult', 2: 'Student', 3: 'Adult', 4: 'Baby', 5: 'Adult', 6: 'Adult'}

for x in range(len(df_train['AgeGroup'])):
    if df_train['AgeGroup'][x] == 'Unknown':
        df_train['AgeGroup'][x] = age_title_mapping[df_train['Title'][x]]
        
for x in range(len(df_test['AgeGroup'])):
    if df_test['AgeGroup'][x] == 'Unknown':
        df_test['AgeGroup'][x] = age_title_mapping[df_test['Title'][x]]
#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
df_train['AgeGroup'] = df_train['AgeGroup'].map(age_mapping)
df_test['AgeGroup'] = df_test['AgeGroup'].map(age_mapping)

print(df_train.head())

#dropping the Age feature for now, might change
df_train = df_train.drop(['Age'], axis = 1)
df_test = df_test.drop(['Age'], axis = 1)
#drop the name feature since it contains no more useful information.
df_train = df_train.drop(['Name'], axis = 1)
df_test = df_test.drop(['Name'], axis = 1)
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
df_train['Sex'] = df_train['Sex'].map(sex_mapping)
df_test['Sex'] = df_test['Sex'].map(sex_mapping)

print(df_train.head())




#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
df_train['Embarked'] = df_train['Embarked'].map(embarked_mapping)
df_test['Embarked'] = df_test['Embarked'].map(embarked_mapping)

df_train.head()

#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(df_test['Fare'])):
    if pd.isnull(df_test['Fare'][x]):
        pclass = df_test['Pclass'][x] #Pclass = 3
        df_test['Fare'][x] = round(df_train[df_train['Pclass'] == pclass]['Fare'].mean(), 4)
#map Fare values into groups of numerical values
df_train['FareBand'] = pd.qcut(df_train['Fare'], 4, labels = [1, 2, 3, 4])
df_test['FareBand'] = pd.qcut(df_test['Fare'], 4, labels = [1, 2, 3, 4])
#drop Fare values
df_train = df_train.drop(['Fare'], axis = 1)
df_test = df_test.drop(['Fare'], axis = 1)
#check train data
print(df_train.head())
#check test data
df_test.head()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

predictors = df_train.drop(['Survived', 'PassengerId'], axis=1)
target = df_train['Survived']
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy via logistic regression =', acc_logreg)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy via Gaussian Naive Bayes =',acc_gaussian)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy via Support Vector Machines =',acc_svc)
# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy via Linear SVC =',acc_linear_svc)
# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy via Perceptron =',acc_perceptron)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy via Decision Tree =',acc_decisiontree)
# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy via Random Forest =',acc_randomforest)
# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy via k-Nearest Neighbors =',acc_knn)
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy via Stochastic Gradient Descent =',acc_sgd)
 #Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy via Gradient Boosting Classifier =',acc_gbk)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
print(models.sort_values(by='Score', ascending=False))
a = models.sort_values(by='Score', ascending=False)
ids = df_test['PassengerId']
gi = df_test.drop('PassengerId', axis=1)
y_pred_new = randomforest.predict(gi)
#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': y_pred_new })
output.to_csv('D:\ML-Program\Titanic/submission.csv', index=False)
'''
#set ids as PassengerId and predict survival 
ids = df_test['PassengerId']
gi = df_test.drop('PassengerId', axis=1)
y_pred_new = randomforest.predict(gi)
#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': y_pred_new })
output.to_csv('D:\ML-Program\Titanic/submission.csv', index=False)'''
