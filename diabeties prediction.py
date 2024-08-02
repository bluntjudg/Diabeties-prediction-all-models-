import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('https://query.data.world/s/ktuy6ss4pdht7iuzn7ccgy5fbhwirk?dws=00000')

mat=df.corr()

sns.heatmap(mat)
plt.show()
print(df.columns)

from sklearn.model_selection import train_test_split


x=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]

x=df[['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

y=df['Outcome']

print(x)

print(y)

x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,train_size=0.95)

print(x_tr)
print(y_tr)

#linear regression 
from sklearn import linear_model
model=linear_model.LogisticRegression()
model.fit(x_tr,y_tr)
pred=model.predict(x_ts)
print('actual out:',list(y_ts))
print('predict out:',pred)
print('accuracy:',model.score(x_tr,y_tr))

#support vector machine 
from sklearn import svm
model=svm.SVC()
model.fit(x_tr,y_tr)
pred=model.predict(x_ts)
print('actual out:',list(y_ts))
print('predict out:',pred)
print('accuracy:',model.score(x_tr,y_tr))


#decisio  tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_tr, y_tr)
pred = model.predict(x_ts)
print('actual out:', list(y_ts))
print('predict out:', pred)
print('accuracy:', model.score(x_tr, y_tr))



#random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion ='entropy', random_state=0) # only 10 trees will b ok as dataset is only 400+ records. Default is 100.    
classifier.fit(x_tr,y_tr)
y_pred = classifier.predict(x_ts)
print('actual out:', list(y_ts))
print('predict out:', pred)
print('accuracy:', model.score(x_tr, y_tr))



#knn - k nearest neighbour
from sklearn import neighbors
model=neighbors.KNeighborsClassifier()
model.fit(x_tr,y_tr)
y_pred=model.predict(x_ts)
print("kNN\nActual output : ",list(y_ts))
print("Predicted output:",y_pred)
print("Accuracy:",model.score(x_tr,y_tr))
