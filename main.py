import pandas as pd

df = pd.read_csv("Student_Performance_Dataset.csv")

print(df.head())
print(df.shape)
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

df.to_csv("cleaned_student_performance.csv", index=False)

print("\nCleaned dataset saved successfully.")


import matplotlib.pyplot as plt
import seaborn as sns

print("\nEDA STARTED")

print("\nPerformance Level:")
print(df["Performance_Level"].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Performance_Level")
plt.title("Performance Level Distribution")
plt.xlabel("Performance Level")
plt.ylabel("Number of Students")
plt.show()

print("\nPass Fail:")
print(df["Pass_Fail"].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Pass_Fail")
plt.title("Pass and Fail Distribution")
plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Final_Percentage", bins=20, kde=True)
plt.title("Final Percentage Distribution")
plt.xlabel("Final Percentage")
plt.ylabel("Number of Students")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Study_Hours_Per_Day",
    y="Final_Percentage"
)
plt.title("Study Hours vs Final Percentage")
plt.xlabel("Study Hours Per Day")
plt.ylabel("Final Percentage")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Attendance_Percentage",
    y="Final_Percentage"
)
plt.title("Attendance vs Final Percentage")
plt.xlabel("Attendance Percentage")
plt.ylabel("Final Percentage")
plt.show()

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nEDA COMPLETED")



features = [
    "Age",
    "Study_Hours_Per_Day",
    "Attendance_Percentage",
    "Parental_Education",
    "Internet_Access",
    "Extracurricular_Activities",
    "Math_Score",
    "Science_Score",
    "English_Score",
    "Previous_Year_Score"
]

X = df[features]
y = df["Final_Percentage"]

print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget Variable:")
print(y.name)

print("\nInput Data Shape:")
print(X.shape)

print("\nTarget Data Shape:")
print(y.shape)

print("\nInput Data:")
print(X.head())

print("\nTarget Data:")
print(y.head())


X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

print("\nEncoded Features:")
print(X_encoded.columns.tolist())

print("\nShape After Encoding:")
print(X_encoded.shape)

print("\nEncoded Data:")
print(X_encoded.head())


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data Shape:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Data Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Data Shape:")
print(X_test_scaled.shape)

print("\nFirst 5 Scaled Training Rows:")
print(X_train_scaled[:5])


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

linear_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor(random_state=42)
random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

linear_model.fit(X_train_scaled, y_train)
decision_tree_model.fit(X_train_scaled, y_train)
random_forest_model.fit(X_train_scaled, y_train)

linear_predictions = linear_model.predict(X_test_scaled)
decision_tree_predictions = decision_tree_model.predict(X_test_scaled)
random_forest_predictions = random_forest_model.predict(X_test_scaled)

print("\nLinear Regression Predictions:")
print(linear_predictions[:10])

print("\nDecision Tree Predictions:")
print(decision_tree_predictions[:10])

print("\nRandom Forest Predictions:")
print(random_forest_predictions[:10])


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_predictions)

decision_tree_mae = mean_absolute_error(y_test, decision_tree_predictions)
decision_tree_mse = mean_squared_error(y_test, decision_tree_predictions)
decision_tree_rmse = np.sqrt(decision_tree_mse)
decision_tree_r2 = r2_score(y_test, decision_tree_predictions)

random_forest_mae = mean_absolute_error(y_test, random_forest_predictions)
random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
random_forest_rmse = np.sqrt(random_forest_mse)
random_forest_r2 = r2_score(y_test, random_forest_predictions)

print("\nLinear Regression Evaluation:")
print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)

print("\nDecision Tree Regression Evaluation:")
print("MAE:", decision_tree_mae)
print("MSE:", decision_tree_mse)
print("RMSE:", decision_tree_rmse)
print("R2 Score:", decision_tree_r2)

print("\nRandom Forest Regression Evaluation:")
print("MAE:", random_forest_mae)
print("MSE:", random_forest_mse)
print("RMSE:", random_forest_rmse)
print("R2 Score:", random_forest_r2)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

classification_features = [
    "Age",
    "Study_Hours_Per_Day",
    "Attendance_Percentage",
    "Parental_Education",
    "Internet_Access",
    "Extracurricular_Activities",
    "Math_Score",
    "Science_Score",
    "English_Score",
    "Previous_Year_Score"
]

X_classification = df[classification_features]

y_classification = df["Performance_Level"]

X_classification = pd.get_dummies(
    X_classification,
    drop_first=True,
    dtype=int
)

label_encoder = LabelEncoder()

y_classification_encoded = label_encoder.fit_transform(
    y_classification
)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_classification,
    y_classification_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_classification_encoded
)

classification_scaler = StandardScaler()

X_train_class_scaled = classification_scaler.fit_transform(
    X_train_class
)

X_test_class_scaled = classification_scaler.transform(
    X_test_class
)

print("\nClassification Target Classes:")
print(label_encoder.classes_)

print("\nClassification Training Shape:")
print(X_train_class_scaled.shape)

print("\nClassification Testing Shape:")
print(X_test_class_scaled.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

logistic_model = LogisticRegression(max_iter=1000, random_state=42)

decision_tree_classifier = DecisionTreeClassifier(
    random_state=42
)

random_forest_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

knn_model = KNeighborsClassifier(n_neighbors=5)

logistic_model.fit(
    X_train_class_scaled,
    y_train_class
)

decision_tree_classifier.fit(
    X_train_class_scaled,
    y_train_class
)

random_forest_classifier.fit(
    X_train_class_scaled,
    y_train_class
)

knn_model.fit(
    X_train_class_scaled,
    y_train_class
)

logistic_predictions = logistic_model.predict(
    X_test_class_scaled
)

decision_tree_class_predictions = decision_tree_classifier.predict(
    X_test_class_scaled
)

random_forest_class_predictions = random_forest_classifier.predict(
    X_test_class_scaled
)

knn_predictions = knn_model.predict(
    X_test_class_scaled
)

print("\nLogistic Regression Predictions:")
print(logistic_predictions[:10])

print("\nDecision Tree Predictions:")
print(decision_tree_class_predictions[:10])

print("\nRandom Forest Predictions:")
print(random_forest_class_predictions[:10])

print("\nKNN Predictions:")
print(knn_predictions[:10])



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

models = {
    "Logistic Regression": logistic_predictions,
    "Decision Tree": decision_tree_class_predictions,
    "Random Forest": random_forest_class_predictions,
    "KNN": knn_predictions
}

for model_name, predictions in models.items():

    accuracy = accuracy_score(
        y_test_class,
        predictions
    )

    precision = precision_score(
        y_test_class,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test_class,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test_class,
        predictions,
       average="weighted",
        zero_division=0
    )

    print("\n" + model_name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test_class,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_class, predictions))


    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

classification_results = []

for model_name, predictions in models.items():

    accuracy = accuracy_score(y_test_class, predictions)

    precision = precision_score(
        y_test_class,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test_class,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test_class,
        predictions,
        average="weighted",
        zero_division=0
    )

    classification_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

classification_results_df = pd.DataFrame(classification_results)

print("\nClassification Model Comparison:")
print(classification_results_df.to_string(index=False))

best_classification_model_name = classification_results_df.loc[
    classification_results_df["F1 Score"].idxmax(),
    "Model"
]

print("\nBest Classification Model:")
print(best_classification_model_name)


new_student = pd.DataFrame({
    "Age": [17],
    "Study_Hours_Per_Day": [5],
    "Attendance_Percentage": [85],
    "Parental_Education": ["Graduate"],
    "Internet_Access": ["Yes"],
    "Extracurricular_Activities": ["Yes"],
    "Math_Score": [75],
    "Science_Score": [72],
    "English_Score": [78],
    "Previous_Year_Score": [70]
})

new_student_encoded = pd.get_dummies(
    new_student,
    drop_first=True,
    dtype=int
)

new_student_encoded = new_student_encoded.reindex(
    columns=X_train_class.columns,
    fill_value=0
)

new_student_scaled = classification_scaler.transform(
    new_student_encoded
)

prediction = logistic_model.predict(
    new_student_scaled
)

predicted_performance = label_encoder.inverse_transform(
    prediction
)

print("\nNew Student Prediction:")
print("Predicted Performance Level:", predicted_performance[0])


import pickle

with open("best_classification_model.pkl", "wb") as file:
    pickle.dump(logistic_model, file)

with open("classification_scaler.pkl", "wb") as file:
    pickle.dump(classification_scaler, file)

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

print("\nBest model saved successfully.")
print("Model: best_classification_model.pkl")
print("Scaler: classification_scaler.pkl")
print("Label Encoder: label_encoder.pkl")