#Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Importing Datasets
df = pd.read_csv('C:\\Users\\prach\\OneDrive\\Desktop\\COLLEGE\\TY\\Machine Learning\\bc_minidata.csv')
x=df.drop("diagnosis", axis=1)
y=df["diagnosis"]
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.2)
z=tuple(y_train)
print(z)
ytest=tuple(y_test)
print(ytest)
print(x_test)

#Using SVM Model
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train,z)
output=tuple(svclassifier.predict(x_test))

#Accuracy Measures
cm=confusion_matrix(ytest, output)
print(cm)
accs=accuracy_score(ytest,output)
print("Accuracy score=", accs)

#Plotting 'area_mean' & 'concave points_mean'
sns.lmplot('area_mean','concave points_mean',df, hue='diagnosis', fit_reg=False)
fig = plt.gcf()
plt.show()