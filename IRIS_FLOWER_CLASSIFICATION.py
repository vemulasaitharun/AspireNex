# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv("IRIS.csv")

# Data preprocessing
# Convert categorical target variable into numerical labels
le = LabelEncoder()
data['species_encoded'] = le.fit_transform(data['species'])

# Splitting into features and target
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species_encoded']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training (Example with Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Cross-validation for further evaluation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")

# Example deployment code if applicable
# (Deployment will depend on your specific deployment environment)

