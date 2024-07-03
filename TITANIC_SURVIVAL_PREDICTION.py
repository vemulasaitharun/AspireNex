import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_titanic_model(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Filling missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data.drop(columns=['Cabin'], inplace=True)  # Dropping 'Cabin' due to many missing values

    # Dropping unnecessary columns
    data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

    # Convert categorical columns to numerical
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

    # Define features and target
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute any remaining missing values
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # Predict on the test set
    y_pred_logreg = logreg.predict(X_test)

    # Evaluate the model
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
    print("Classification Report:\n", classification_report(y_test, y_pred_logreg))

    return logreg, imputer, scaler

# Train the model
model, imputer, scaler = train_titanic_model('titanic.csv')




def predict_survival(model, imputer, scaler, new_data):
    # Convert new data to DataFrame
    new_data_df = pd.DataFrame(new_data)

    # Handle missing values
    new_data_imputed = imputer.transform(new_data_df)

    # Feature scaling
    new_data_scaled = scaler.transform(new_data_imputed)

    # Predict the survival chance
    survival_prediction = model.predict(new_data_scaled)
    survival_prob = model.predict_proba(new_data_scaled)

    return survival_prediction, survival_prob

# Example new passenger data
new_passenger = {
    'Pclass': [3],
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [7.25],
    'Sex_male': [1],  # 1 if male, 0 if female
    'Embarked_Q': [0],
    'Embarked_S': [1]
}

# Predict survival
prediction, probability = predict_survival(model, imputer, scaler, new_passenger)

print(f"Survival Prediction: {prediction[0]}")
print(f"Survival Probability: {probability[0][1]:.2f}")
