import pandas as pd
data = pd.read_csv('../input/train.csv')

#data.head()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16,8))
sns.heatmap(data.corr(),annot=True,linewidths=1,linecolor='w')
plt.xlabel('Columns')
plt.ylabel('Columns')
plt.title('Heatmap')
plt.savefig('Heatmap.png')

training_data=data[['PassengerId','Pclass','Fare']]
output_data=data['Survived']

from sklearn.model_selection import train_test_split as tts
X,x_test,Y,y_test= tts(training_data,output_data,test_size=0.3,random_state=42)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X,Y)
predictions=model.predict(x_test)
predictions[:5]

from sklearn.metrics import accuracy_score, confusion_matrix
score = accuracy_score(predictions,y_test)
print (score)

test_data=pd.read_csv('../input/test.csv')
test_data.head()

test_data=test_data[['PassengerId','Pclass','Fare']]
test_data.head()

test_data.isnull().sum()

test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())

test_data.isnull().sum()

test_predictions=model.predict(test_data)

submission=pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':test_predictions})

submission.head()

submission=submission.set_index('PassengerId')

submission.to_csv('Prediction1.csv')

data['Sex']=data['Sex'].apply(lambda x:1 if x=='male' else 0)
data.head()
td=data[['PassengerId','Pclass','Fare','Sex']]
od=data['Survived']

td.head()
X,x_test,Y,y_test= tts(td,od,test_size=0.3,random_state=42)
model=LogisticRegression()
model.fit(X,Y)
predictions=model.predict(x_test)
predictions[:5]

score = accuracy_score(predictions,y_test)
print (score)

test_data=pd.read_csv('../input/test.csv')
test_data.head()

test_data['Sex']=test_data['Sex'].apply(lambda x:1 if x=='male' else 0)
test_data.head()

test_data=test_data[['PassengerId','Pclass','Fare','Sex']]
#test_data.head()
test_data.isnull().sum()
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())
test_data.isnull().sum()

test_predictions=model.predict(test_data)

submission=pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':test_predictions})

#submission.head()














