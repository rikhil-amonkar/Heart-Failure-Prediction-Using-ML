import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask

# Load the dataset
heart_data = pd.read_csv("/Users/rikhilamacpro/Downloads/heart.csv")

# Label encoding for categorical features
encoder = LabelEncoder()
heart_data['Sex'] = encoder.fit_transform(heart_data['Sex'])
heart_data['ChestPainType'] = encoder.fit_transform(heart_data['ChestPainType'])
heart_data['RestingECG'] = encoder.fit_transform(heart_data['RestingECG'])
heart_data['ExerciseAngina'] = encoder.fit_transform(heart_data['ExerciseAngina'])
heart_data['ST_Slope'] = encoder.fit_transform(heart_data['ST_Slope'])

# Drop rows with missing data
heart_data = heart_data.dropna()

# Fill missing data for numerical features with mean values
heart_data['Age'] = heart_data['Age'].fillna(heart_data['Age'].mean())
heart_data['Sex'] = heart_data['Sex'].fillna(heart_data['Sex'].mean())
heart_data['ChestPainType'] = heart_data['ChestPainType'].fillna(heart_data['ChestPainType'].mean())
heart_data['RestingBP'] = heart_data['RestingBP'].fillna(heart_data['RestingBP'].mean())
heart_data['Cholesterol'] = heart_data['Cholesterol'].fillna(heart_data['Cholesterol'].mean())
heart_data['FastingBS'] = heart_data['FastingBS'].fillna(heart_data['FastingBS'].mean())
heart_data['RestingECG'] = heart_data['RestingECG'].fillna(heart_data['RestingECG'].mean())
heart_data['MaxHR'] = heart_data['MaxHR'].fillna(heart_data['MaxHR'].mean())
heart_data['ExerciseAngina'] = heart_data['ExerciseAngina'].fillna(heart_data['ExerciseAngina'].mean())
heart_data['Oldpeak'] = heart_data['Oldpeak'].fillna(heart_data['Oldpeak'].mean())
heart_data['ST_Slope'] = heart_data['ST_Slope'].fillna(heart_data['ST_Slope'].mean())
heart_data['HeartDisease'] = heart_data['HeartDisease'].fillna(heart_data['HeartDisease'].mean())

# Feature selection and target variable
features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'Oldpeak', 'MaxHR', 'ExerciseAngina']
X = heart_data[features]
y = heart_data.HeartDisease

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32, random_state=42)

# Initialize the StandardScaler and fit it on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the fitted scaler
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
heart_disease_model = LogisticRegression(random_state=0, max_iter=1000)
heart_disease_model.fit(X_train_scaled, y_train)

# Make predictions
predict_failure = heart_disease_model.predict(X_test_scaled)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, predict_failure)
accuracy_percentage = round(accuracy, 3) * 100
print("This Heart Disease Predictor Model's Accuracy: %" + str(accuracy_percentage))

# Test patients with predefined data
test_patients = [
    {'Age': 52, 'Sex': 1, 'ChestPainType': 0, 'RestingBP': 140, 'Cholesterol': 250, 'FastingBS': 0, 'RestingECG': 1, 'MaxHR': 160, 'ExerciseAngina': 0, 'Oldpeak': 0.5, 'ST_Slope': 1},
    {'Age': 45, 'Sex': 0, 'ChestPainType': 2, 'RestingBP': 130, 'Cholesterol': 230, 'FastingBS': 0, 'RestingECG': 2, 'MaxHR': 150, 'ExerciseAngina': 1, 'Oldpeak': 1.2, 'ST_Slope': 2},
    {'Age': 60, 'Sex': 1, 'ChestPainType': 1, 'RestingBP': 120, 'Cholesterol': 300, 'FastingBS': 1, 'RestingECG': 1, 'MaxHR': 110, 'ExerciseAngina': 1, 'Oldpeak': 2.0, 'ST_Slope': 2},
    {'Age': 38, 'Sex': 0, 'ChestPainType': 1, 'RestingBP': 128, 'Cholesterol': 180, 'FastingBS': 0, 'RestingECG': 1, 'MaxHR': 170, 'ExerciseAngina': 0, 'Oldpeak': 0.3, 'ST_Slope': 1},
    {'Age': 49, 'Sex': 0, 'ChestPainType': 2, 'RestingBP': 160, 'Cholesterol': 180, 'FastingBS': 0, 'RestingECG': 1, 'MaxHR': 156, 'ExerciseAngina': 0, 'Oldpeak': 1.0, 'ST_Slope': 1}
]

# Predict heart disease for test patients
for case in test_patients:
    user_feature_df = pd.DataFrame([case], columns=features)
    user_feature_scaled = scaler.transform(user_feature_df)  # Scale the user input
    probabilities = heart_disease_model.predict_proba(user_feature_scaled)
    heart_disease_probability = round(probabilities[0][1], 3) * 100
    print(f"User input: {case}")
    print(f"The probability of this patient having heart disease is: {heart_disease_probability:.2f}%")
    if heart_disease_probability >= 60:
        print("You should immediately consult with a health professional as your chance of heart disease is predicted to be very high.")
    elif heart_disease_probability > 20 and heart_disease_probability < 60:
        print("You might need to make some lifestyle changes or possibly consult with a health professional as your chance of heart disease is higher than normal.")
    else:
        print("You are living a very healthy lifestyle. Keep it up! Your chances of heart disease are very low.")
