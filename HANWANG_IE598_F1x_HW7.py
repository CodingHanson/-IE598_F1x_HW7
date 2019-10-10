import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time

df = pd.read_csv('F:/MSFE/machine_learning/HW7/ccdefault.csv')
y = df['DEFAULT']
X = df.drop(['ID','DEFAULT'],axis=1)
print('X:',X.head(),'y:',y.head())
feat_labels = df.columns[1:]
print(feat_labels)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=16)

from sklearn.model_selection import cross_val_score

N = [50,100,200,300,400,500,600,700,800,900,1000]

time_list = []
scores = []
for i in N:
    start = time.clock()
    rf = RandomForestClassifier(n_estimators= i,random_state=16)
    rf.fit(X_train,y_train)
    scores.append(np.mean(cross_val_score(estimator=rf, X=X_train, y=y_train, cv=10, n_jobs=4)))
    end = time.clock()
    time_list.append(end-start)

best_score = max(scores)
best_n_estimator = N[scores.index(best_score)]
cost_time = time_list[scores.index(best_score)]
print("best score is :", best_score, " its n_estimator is : ", best_n_estimator,
      ", and it take",cost_time,"s to run")
for i in range(len(scores)):
    print("n_estimators:",N[i],", scores: ",scores[i],", time taken",time_list[i],"s" )

import matplotlib.pyplot as plt
forest = RandomForestClassifier(n_estimators=best_n_estimator,random_state=1)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d %-*s %f" % (f+1,30,feat_labels[indices[f]],importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels,rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()