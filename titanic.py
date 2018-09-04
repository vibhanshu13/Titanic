# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:09:23 2018

@author: vibhanshuv
"""

import pandas as pd
import numpy as np

train_df=pd.read_csv('train.csv',index_col='PassengerId')
test_df=pd.read_csv('test.csv',index_col='PassengerId')

type(train_df)
train_df.info()
test_df.info()

test_df['Survived']=-58
df=pd.concat((train_df,test_df))

male_passengers=df.loc[df.Sex=='male',:]
print('number of male passengers : {0}'.format(len(male_passengers)))

male_passengers_first_class=df.loc[((df.Sex=='male') & (df.Pclass==1))]
print('number of male passengers in first class : {0}'.format(len(male_passengers_first_class)))

df.describe()

print('mean fare :{0}'.format(df.Fare.mean()))
print('median fare :{0}'.format(df.Fare.median()))
print('min fare :{0}'.format(df.Fare.min()))
print('max fare :{0}'.format(df.Fare.max()))
print('Fare range :{0}'.format(df.Fare.max()-df.Fare.min()))
print('25% Fare quantile :{0}'.format(df.Fare.quantile(0.25)))
print('50% Fare quantile :{0}'.format(df.Fare.quantile(0.5)))
print('75% Fare quantile :{0}'.format(df.Fare.quantile(0.75)))
print('fare variance :{0}'.format(df.Fare.var()))
print('fare standard deviation :{0}'.format(df.Fare.std()))

df.Fare.plot(kind='box')

df.describe(include='all')

df.Sex.value_counts(normalize=True)

df[df.Survived !=-58].Survived.value_counts()


df.Pclass.value_counts().plot(kind='bar',rot=0, title='class wise passenger details')

#Univariate plots
df.Age.plot(kind='hist',title='histogram for age',color='c',bins=20)

df.Age.plot(kind='kde',title='density plot for age',color='c')

df.Fare.plot(kind='hist',title='histogram for Fare',color='c',bins=20)

print('skewness for fare :{0}'.format(df.Fare.skew()))

print('skewness for Age :{0}'.format(df.Age.skew()))

#bivariate plot
df.plot.scatter(x='Age',y='Fare',color='c',title='scatter plot: Age vs Fare')


df.plot.scatter(x='Age',y='Fare',color='c',title='scatter plot: Age vs Fare',alpha=0.1)

df.plot.scatter(x='Pclass',y='Fare',color='c',title='scatter plot: Pclass vs Fare',alpha=0.15)

# groupby function
df.groupby('Sex').Age.median()
df.groupby('Pclass').Fare.median()

df[['Pclass','Age','Fare']].groupby('Pclass').mean()

# cross tab
pd.crosstab(df.Sex,df.Pclass)
pd.crosstab(df.Sex,df.Pclass).plot(kind='bar')

#Pivot table

df.pivot_table(index='Sex',columns='Pclass',values='Age',aggfunc='mean')

#Treating missing values
df.info()

#Filling embarked
df[df.Embarked.isnull()]

df.Embarked.value_counts()
pd.crosstab(df[df.Survived!=-58].Survived,df[df.Survived!=-58].Embarked)
#df.loc[df.Embarked.isnull(),'Embarked']=='S'
df.Embarked.fillna('S',inplace=True)
df.roupby(['Pclass','Embarked']).Fare.median()
df.Embarked.fillna('C',inplace=True)

df.info()

# Filling Fare
df[df.Fare.isnull()]
median_f=df.loc[(df.Pclass==3)&(df.Embarked=='S'),'Fare'].median()
df.Fare.fillna(median_f,inplace=True)

df.info()

#Filling age
pd.options.display.max_rows=15
df[df.Age.isnull()]

df.Age.plot(kind='hist',bins=20,color='c')
df.Age.mean()
df.groupby('Sex').Age.median()

df[df.Age.notnull()].boxplot('Age','Sex')

df.groupby('Pclass').Age.median()

df.Name



def GetTitle(name):
    title_group={'mr':'Mr',
                 'mrs':'Mrs',
                 'master':'Master',
                 'miss':'Miss',
                 'don':'Sir',
                 'rev':'Sir',
                 'dr':'Officer',
                 'mme':'Mrs',
                 'ms':'Mrs',
                 'major':'Officer',
                 'lady':'Lady',
                 'sir':'Sir',
                 'mlle':'Miss',
                 'col':'Officer',
                 'capt':'Officer',
                 'the countess':'lady',
                 'jonkheer':'Sir',
                 'dona':'Lady'}
    first_name_with_title=name.split(',')[1]
    title=first_name_with_title.split('.')[0]
    title=title.strip().lower()
    return title_group[title]    

df['Title']=df.Name.map(lambda x :GetTitle(x))
df[df.Age.notnull()].boxplot('Age','Title');

title_age_median=df.groupby('Title').Age.transform('median')
df.Age.fillna(title_age_median, inplace=True)

df.info()

#Treating outliers
df.Age.plot(kind='hist',bins=20,color='c')

df.loc[df.Age>70]
df.Fare.plot(kind='hist',title='histogram for fare',bins=20,color='r')

df.Fare.plot(kind='box')
df.loc[df.Fare==df.Fare.max()]

#Try some transformations  to reduce the skewness
LogFare=np.log(df.Fare+1)
LogFare.plot(kind='Hist',bins=20,color='r')

#binning
pd.qcut(df.Fare,4)

pd.qcut(df.Fare,4, labels=['very low','low','high','very high'])


pd.qcut(df.Fare,4, labels=['very low','low','high','very high']).value_counts().plot(kind='bar',color='r',rot=0)

df['Fare_bin']=pd.qcut(df.Fare,4, labels=['very low','low','high','very high'])

# feature engineering
df['AgeState']=np.where(df['Age']>=18,'Adult','Child')
df['AgeState'].value_counts()
pd.crosstab(df[df.Survived!=-58].Survived,df[df.Survived!=-58].AgeState)

df['FamilySize']=df.Parch+df.SibSp+1
df['FamilySize'].plot(kind='hist',bins=20,color='r')

df.loc[df.FamilySize==df.FamilySize.max(),['Name','Survived','FamilySize','Ticket']]
pd.crosstab(df[df.Survived!=-58].Survived,df[df.Survived!=-58].FamilySize)

df['Ismother']=np.where(((df.Sex=='female')& (df.Parch>0) & (df.Age>18) & (df.Title !='Miss')),1,0)
pd.crosstab(df[df.Survived!=-58].Survived,df[df.Survived!=-58].Ismother)

df.Cabin
df.Cabin.unique()
df.loc[df.Cabin=='T']

df.loc[df.Cabin=='T','Cabin']=np.nan

df.Cabin.unique()

def getdeck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')
df['Deck']=df['Cabin'].map(lambda x :getdeck(x ))

df.Deck.value_counts()
pd.crosstab(df[df.Survived!=-58].Survived,df[df.Survived!=-58].Deck)

df.info()

# Converting to numeric values

df['Ismale']=np.where(df.Sex=='male',1,0)
df=pd.get_dummies(df,columns=['Deck','Pclass','Title','Fare_bin','Embarked','AgeState'])

df.info()

df.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis=1,inplace=True)
columns=[column for column in df.columns if column!='Survived']
columns=['Survived']+columns
df=df[columns]

df.info()

# separating train and test data
df.loc[df.Survived !=-58].to_csv('tr.csv')

columns=[column for column in df.columns if column!='Survived']
df.loc[df.Survived==-58,columns].to_csv('te.csv')#index=False



traindf=pd.read_csv('tr.csv',index_col='PassengerId')
testdf=pd.read_csv('te.csv',index_col='PassengerId')

X=traindf.loc[:,'Age':].as_matrix().astype('float')
Xtest=testdf.loc[:,'Age':].as_matrix().astype('float')
y=train_df['Survived'].ravel()

print (X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Checking imbalance
print('mean survival in train :{0:.3F}'.format(np.mean(Y_train))) 

print('mean survival in test :{0:.3F}'.format(np.mean(Y_test)))

import sklearn
sklearn.__version__


from sklearn.dummy import DummyClassifier
md=DummyClassifier(strategy='most_frequent',random_state=0)
md.fit(X_train,Y_train)
md.score(X_test,Y_test)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

print('Accuracy score:{0:.3F}'.format(accuracy_score(Y_test,md.predict(X_test))))

print('confusion matrix:{0}'.format(confusion_matrix(Y_test,md.predict(X_test))))

print('Precision score:{0:.3F}'.format(precision_score(Y_test,md.predict(X_test))))

print('Recall score:{0:.3F}'.format(recall_score(Y_test,md.predict(X_test))))


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
from sklearn.model_selection import GridSearchCV
parameters={'C':[1.0,10.,50.0,100.0,500.0,1000.0],'penalty':['l1','l2']}
clf=GridSearchCV(model,param_grid=parameters,cv=3)
clf.fit(X_train,Y_train)
clf.best_params_
print('best score :{0:.2F}'.format(clf.best_score_))
clf.score(X_test,Y_test)
clf.fit(X,y)
ss=clf.predict(Xtest)


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=20,metric='minkowski')
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=500)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.svm import SVC
model = SVC(kernel='rbf', C=100)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
from sklearn.model_selection import GridSearchCV
parameters={'n_estimators':[1.0,10.,50.0,100.0,500.0,1000.0]}
clf=GridSearchCV(model,param_grid=parameters,cv=3)
clf.fit(X_train,Y_train)
clf.best_params_
print('best score :{0:.2F}'.format(clf.best_score_))

model.fit(X,y)
ss=model.predict(Xtest)


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
model.fit(X,y)
ss=model.predict(Xtest)


col=['PassengerId','Survived']
df_csv = pd.DataFrame(columns=col)
df_csv.PassengerId=test_df.index
df_csv.Survived=ss
df_csv.to_csv('Titanic_survivor2.csv',index=False)