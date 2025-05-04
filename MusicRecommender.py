#Import Required Libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

#Load the Dataset:
df = pd.read_csv('music.csv')

#Explore the Data:
df.head()  # View first 5 rows
df.describe()  # Statistical summary
print(df.shape)  # Expected output: (18, 3)

#Prepare the Data:
X = df.drop(columns=['genre'])
y = df['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Model:
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#Evaluate the Model:
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

#Save the Model:
joblib.dump(model, 'music_recommender.joblib')







