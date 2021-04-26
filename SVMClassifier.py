import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

df = pd.read_csv("Copy.csv")
valid_test,test_set, train_set= np.split(df, [int(.0445*len(df)),int(.0896*len(df))])
test_set.to_csv("my.csv", encoding='utf-8')
y_test = test_set['Winners']
y_train = train_set['Winners']
X_train = train_set[['Year','Team1FGA','Team1FG%','Team13PA','Team13P%','Team1REB','Team1TOV','Team1STL','Team2FGA','Team2FG%','Team23PA','Team23P%','Team2REB','Team2TOV','Team2STL']]
X_test = test_set[['Year','Team1FGA','Team1FG%','Team13PA','Team13P%','Team1REB','Team1TOV','Team1STL','Team2FGA','Team2FG%','Team23PA','Team23P%','Team2REB','Team2TOV','Team2STL']]

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

clf = SVC(gamma='scale',probability=True)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

y_pred = clf.predict(X_test)

pd.DataFrame(y_pred).to_csv("final.csv")

print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print(cm)

