import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

df = pd.read_csv("Copy.csv")

valid_test,test_set, train_set= np.split(df, [int(.0445*len(df)),int(.0896*len(df))])
y_test = test_set['Winners']
y_train = train_set['Winners']
X_train = train_set[['Year','Team1FGA','Team1FG%','Team13PA','Team13P%','Team1REB','Team1TOV','Team1STL','Team2FGA','Team2FG%','Team23PA','Team23P%','Team2REB','Team2TOV','Team2STL']]
X_test = test_set[['Year','Team1FGA','Team1FG%','Team13PA','Team13P%','Team1REB','Team1TOV','Team1STL','Team2FGA','Team2FG%','Team23PA','Team23P%','Team2REB','Team2TOV','Team2STL']]

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

rdf = RandomForestClassifier(max_depth=8, random_state=0, n_estimators=300)
rdf.fit(X_train, y_train)
rdf.score(X_test,y_test)

y_pred = rdf.predict(X_test)

pd.DataFrame(y_pred).to_csv("final.csv")

print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print(cm)
