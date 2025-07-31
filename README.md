# ML Project: Employee Attrition Prediction

Part of my internship project at Devstringx 

##  Objective
Predict whether an employee will leave the company (Attrition = Yes) using machine learning, based on HR data like age, salary, department, job role, years at company, etc.

## How the project works (step by step)

**Step 1: Load data**  
- Used pandas to read the CSV file `Employee_Attrition_Dataset.csv`.
- Explored data shape, columns, and types to understand what’s inside.

**Step 2: Data cleaning & preprocessing**  
- Dropped irrelevant columns: `EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`, `StockOptionLevel`.
- Filled missing values using most frequent value with SimpleImputer.
- Encoded categorical text columns (like Department, Gender, OverTime) into numbers using LabelEncoder.
- Scaled numeric columns (`Age`, `MonthlyIncome`, `YearsAtCompany`, `DistanceFromHome`) with StandardScaler to help the models learn better.

**Step 3: Exploratory Data Analysis (EDA)**  
- Count plot to see how many employees left vs stayed.
- Correlation heatmap to find which features relate to attrition.

**Step 4: Train-Test Split**  
- Split data: 80% for training (to teach the model), 20% for testing (to see if it learned well).

**Step 5: Model Training**  
- Trained two models: Logistic Regression and Decision Tree Classifier.
- Both models learned from training data to predict if an employee might leave.

**Step 6: Evaluation**  
- Tested models on test data.
- Used metrics: Accuracy (overall correct), Precision (correct “leave” predictions), Recall (caught actual leavers), F1 Score (balance).
- Plotted confusion matrix to see where models made mistakes.


# How it works 
We show the model examples of employees who stayed or left.  
It learns patterns (like low salary + high distance might mean leaving).  
Then we ask the model to guess for new employees, and check how good it was.


# Tech stack
- Python, pandas, scikit-learn, seaborn, matplotlib


# About me
**Sugam**  
- B.Tech in Information Technology  
- Data Analytics Intern at Devstringx  
- Passionate about data science & analytics  

