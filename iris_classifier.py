import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
print("Loading data from CSV...")
data = pd.read_csv('iris.csv').drop('Id', axis=1)
print("\nHere is what your data looks like:")
print(data.head())
X = data.iloc[:, :-1] 
y = data.iloc[:, -1]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nSuccess! Final Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed breakdown of how well it predicted each species:")
print(classification_report(y_test, predictions))