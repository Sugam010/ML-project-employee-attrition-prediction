# Predict Employee Attrition ML Project
# Dataset: IBM HR Analytics Employee Attrition & Performance

# Step 1: Load the dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
df = pd.read_csv('Employee_Attrition_Dataset.csv')

# Display basic info
print("Shape of data:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData info:")
print(df.info())

# Step 2: Data cleaning & preprocessing

# 2.1 Drop irrelevant columns
df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber', 'StockOptionLevel'], axis=1, inplace=True)
print("\nColumns after dropping irrelevant ones:")
print(df.columns)

# 2.2 Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# 2.3 Fill missing values if any
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print("\nAny missing values now:")
print(df.isnull().sum())

# 2.4 Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])
print("\nData after encoding:")
print(df.head())

# 2.5 (Optional) Scale numeric columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("\nData after scaling:")
print(df.head())

# Step 3: Exploratory Data Analysis (EDA)

# Plot count of Attrition
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title('Count of Attrition (0=No, 1=Yes)')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 4: Train-Test Split
from sklearn.model_selection import train_test_split
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Step 5: Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Step 6: Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Predict
log_preds = log_model.predict(X_test)
tree_preds = tree_model.predict(X_test)

print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, log_preds))
print("Precision:", precision_score(y_test, log_preds))
print("Recall:", recall_score(y_test, log_preds))
print("F1 Score:", f1_score(y_test, log_preds))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, log_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

print("\n=== Decision Tree Results ===")
print("Accuracy:", accuracy_score(y_test, tree_preds))
print("Precision:", precision_score(y_test, tree_preds))
print("Recall:", recall_score(y_test, tree_preds))
print("F1 Score:", f1_score(y_test, tree_preds))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, tree_preds), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

print("\n✅ Project completed successfully!")
