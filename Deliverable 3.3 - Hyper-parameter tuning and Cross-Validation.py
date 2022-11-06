#################### Hyper Parameter Tuning ####################

#Step-1: Reading the data

import pandas as pd

df=pd.read_csv(r"C:\360\Data_feature_engineered.csv")

#Step-2: Train and Test Split

from sklearn.model_selection import train_test_split
import numpy as np

X=df.iloc[:,0:-1]
Y=df["Renewal"]

#Considering the 30% for test dataset
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
##### Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklear.metrics import accuracy_score,confusion_matrix
par={"n_estimators":[100,200,300,400],"max_depth":[5,10,20,30,50],"min_samples_split":[5,10,20]}
a=RandomForestClassifier()
m1=GridSearchCV(a,param_grid=par,cv=5)
m1.fit(X_train,Y_train)
gds=m1.best_estimator_
GS=gds.fit(X_train,Y_train)

# Evaluation on Training Data
print(confusion_matrix(Y_train, GS.predict(X_train)))
GS_train_acc = accuracy_score(GS.predict(X_train),Y_train)
print("Grid Search Train Accuracy: ","{:.2%}".format(GS_train_acc))

# Evaluation on Testing Data
print(confusion_matrix(Y_test,GS.predict(X_test)))
GS_test_acc = accuracy_score(GS.predict(X_test),Y_test)
print("Grid Search Test Accuracy: ","{:.2%}".format(GS_test_acc))


'''Since GridSearch-Cross_Validation uses K-Fold 
there is no need to create seperate Train and Test data,
Predictors and Target of full dataset will be passed'''
from sklearn.model_selection import GridSearchCV
#decalring a model instance
knn = KNeighborsClassifier()
svm = SVC()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
lr = LogisticRegression()
gnb = GaussianNB()

#creating list of models
model_all = [knn , svm , rf , dt , lr , gnb]

#decalring parameters for hyper tuning
param1 = {"n_neighbors":[3,5,7,9,11,13,15,17,19,21],
          "weights":['uniform','distance'],
          "metric":['euclidean','manhattan']}

param2 = {'C': [0.1,1, 10, 100],
          'gamma': [1,0.1,0.01,0.001],
          'kernel': ['rbf', 'poly', 'sigmoid']}
param3 = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

param4 = { 'criterion':['gini','entropy'],
          'max_depth': np.arange(3, 15)}

param5 = {'penalty': ['l1', 'l2'],
          'C':[0.001,.009,0.01,.09,1,5,10,25]}

param6 = {
    'var_smoothing': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
}

#creating list of parameters
model_param =[param1, param2, param3, param4 , param5 , param6]

#Creating a string names for all the models, later used to campare model performance
model_log = ["_knn", "_svm", "_rf", "_dt" , "_lr" , "_gnb"]

#creating empty df
Grid_knn = Grid_svm = Grid_rf = Grid_dt = Grid_lr = Grid_gnb = pd.DataFrame()

#######################GridSearch-Cross_Validation function#################
#Part-7
#creating k-fold of 10
for i in range(len(model_all)):
    Grid=GridSearchCV(estimator=model_all[i], param_grid=model_param[i], 
                      n_jobs=-1, cv=10, verbose=3 ).fit(X,Y)
    globals()['Grid%s' % model_log[i]]=pd.DataFrame(Grid.cv_results_)

    '''above loop creates dataframes for all possible combinations of models having columns named mean test scores 
and ranking(ranking for mean test scores) '''

#selecting rank 1 model in each dataset
best_knn = Grid_knn[['mean_test_score' , 'rank_test_score']].query('rank_test_score== 1')
best_svm = Grid_svm[['mean_test_score' , 'rank_test_score']].query('rank_test_score== 1')
best_rf = Grid_rf[['mean_test_score' , 'rank_test_score']].query('rank_test_score== 1')
best_dt = Grid_dt[['mean_test_score' , 'rank_test_score']].query('rank_test_score== 1')
best_lr = Grid_lr[['mean_test_score' , 'rank_test_score']].query('rank_test_score== 1')
best_gnb = Grid_gnb[['mean_test_score' , 'rank_test_score']].query('rank_test_score== 1')

#printing test accuracy for all best models
print("Test accuracy for best KNeighborsClassifier model:",format(100*best_knn.iloc[0,0],".2f"),"%")
print("Test accuracy for best SVC model:",format(100*best_svm.iloc[0,0],".2f"),"%")
print("Test accuracy for best RandomForestClassifier model:",format(100*best_rf.iloc[0,0],".2f"),"%")
print("Test accuracy for best DecisionTreeClassifier model:",format(100*best_dt.iloc[0,0],".2f"),"%")
print("Test accuracy for best LogisticRegression model:",format(100*best_lr.iloc[0,0],".2f"),"%")
print("Test accuracy for best GaussianNB model:",format(100*best_gnb.iloc[0,0],".2f"),"%")

# Models with their respective Accuracy


from sklearn.model_selection import cross_val_score

scores=cross_val_score(LogisticRegression(), X, Y, cv=10)
print(min(scores))
print(max(scores))
print(scores.mean())