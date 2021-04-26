import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE

df = pd.read_csv("Copy.csv")

#Pearson Correlation (Filter Method) 
'''
plt.figure(figsize=(20,18))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
'''

#(Wrapper Method)
X = df.drop("Winners",1)   #Feature Matrix
y = df["Winners"]

X_1 = sm.add_constant(X)
#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
print(model.pvalues)

print(df.Winners[df.Winners==0].count())

valid_test,test_set, train_set= np.split(df, [int(.0445*len(df)),int(.0896*len(df))])
y_test = test_set['Winners']
y_train = train_set['Winners']
X_train = train_set[['Year','Team1FGA','Team1FG%','Team13PA','Team1REB','Team1TOV','Team1STL','Team2FGA','Team2FG%','Team23PA','Team2REB','Team2TOV','Team2STL']]
X_test = test_set[['Year','Team1FGA','Team1FG%','Team13PA','Team1REB','Team1TOV','Team1STL','Team2FGA','Team2FG%','Team23PA','Team2REB','Team2TOV','Team2STL']]

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=37)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

pd.DataFrame(y_pred).to_csv("final.csv")


print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print(cm)

#Find the best K

'''
for i in range(1,40):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    error_rate.append(np.mean(y_pred != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
input('press <ENTER> to continue')
'''
