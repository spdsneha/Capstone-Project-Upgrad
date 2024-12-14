# Anoma-Data-Project
AnomaData: Predictive Maintenance Solution utilizing machine learning for automated anomaly detection in equipment, focusing on data exploration, preprocessing, and logistic regression modeling.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_excel('C:/Users/user/Downloads/AnomaData.xlsx')
data.dtypes
data.describe()
missing_values = data.isnull().sum()
missing_values_percentage = (missing_values / len(data)) * 100
missing_values_numeric = missing_values.astype(int)
# Handling missing values
# Drop rows with missing values
data.dropna(inplace=True)
missing_values = data.isnull().sum()
# Handling outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
  # Apply remove_outliers function to each numeric column
numeric_columns = data.select_dtypes(include=['number']).columns
for col in numeric_columns:
    data = remove_outliers(data, col)
  # Standardization
def standardize(df):
    df_std = df.copy()
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    return df_std
data_scaled = standardize(data.drop('y', axis=1))
# Concatenate scaled features with target column
cleaned_data = pd.concat([data_scaled, data['y']], axis=1)
#what we did here?
#1. We first load the dataset.
#2. Then, we handle missing values by dropping rows with missing values.
#3. Next, we handle outliers using the IQR (Interquartile Range) method. This method calculates the first quartile (Q1), third quartile (Q3), and interquartile range (IQR) for each numeric column, and then removes rows where the values fall outside the range [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
#4. After that, we perform standardization by subtracting the mean and dividing by the standard deviation for each feature. This standardization process is performed manually without using StandardScaler.
#5. Finally, we concatenate the standardized features with the target column to obtain the cleaned dataset
data = pd.read_excel('C:/Users/user/Downloads/AnomaData.xlsx')
# Feature Engineering: Creating new features
# Example 1: Aggregating features
# Sum of x1 to x52
data['sum_x'] = data.iloc[:, 2:53].sum(axis=1) 
data['sum_x'].head()

plt.hist(data['sum_x'], bins=20)  # Adjust the number of bins as needed
plt.xlabel('Sum of x1 to x52')
plt.ylabel('Frequency')
plt.title('Distribution of Sum of x1 to x52')
plt.show()
#First 5 sum values
data['sum_x'].head()
# Find the row with the highest sum value
highest_sum_row = data.loc[data['sum_x'].idxmax()]

# Find the index of the row with the highest sum value
highest_sum_index = data['sum_x'].idxmax()

# Extract the row with the highest sum value
highest_sum_row = data.loc[highest_sum_index, ['time', 'sum_x']]
print("Row with the highest sum value:")
highest_sum_row
# Mean of x1 to x52
data['mean_x'] = data.iloc[:, 2:53].mean(axis=1)  
data['mean_x'].head()
#ploting the [mean_x]
plt.figure(figsize=(10, 6))
plt.bar(data.index, data['mean_x'], color='skyblue')
plt.title('Mean of x1 to x52')
plt.xlabel('Index')
plt.ylabel('Mean Value')
plt.grid(axis='y')
plt.show()
#ploting the first 5 [mean_x]
data_subset = data.head(5)

# Create a bar chart for mean_x data of the first 5 rows
plt.figure(figsize=(8, 5))  # Set the figure size
bars = plt.bar(data_subset.index, data_subset['mean_x'], color='skyblue')

# Add value annotations on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, round(height, 2), ha='center', va='bottom')

plt.title('Mean of x1 to x52 (First 5 Rows)')
plt.xlabel('Index')
plt.ylabel('Mean Value')
plt.grid(axis='y')
plt.xticks(data_subset.index)
plt.show()
# Max value of x1 to x52
data['max_x'] = data.iloc[:, 2:53].max(axis=1)
data['max_x'].head()
# Example 2: Feature transformation
 # Square of x1
data['x1_squared'] = data['x1'] ** 2
data['x1_squared'].head()
# Example 3: Date features (Because my 'time' column is in datetime format)
data['hour'] = data['time'].dt.hour
data['date'] = data['time'].dt.date
data[['time', 'hour', 'date']]
#What we did here?
#data['sum_x'] = data.iloc[:, 2:53].sum(axis=1): This line calculates the sum of values across columns x1 to x52 for each row in the DataFrame data and stores the result in a new column named 'sum_x'.
#data['mean_x'] = data.iloc[:, 2:53].mean(axis=1): This line calculates the mean (average) of values across columns x1 to x52 for each row in the DataFrame data and stores the result in a new column named 'mean_x'.
#data['max_x'] = data.iloc[:, 2:53].max(axis=1): This line finds the maximum value across columns x1 to x52 for each row in the DataFrame data and stores the result in a new column named 'max_x'.
#data['min_x'] = data.iloc[:, 2:53].min(axis=1): This line finds the minimum value across columns x1 to x52 for each row in the DataFrame data and stores the result in a new column named 'min_x'.
#data['x1_squared'] = data['x1'] ** 2: This line squares the values in the column 'x1' for each row in the DataFrame data and stores the result in a new column named 'x1_squared'.
#Lastly I showed my unstructured date and time format in a proper date and time format.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#1st I load the data again
data = pd.read_excel('C:/Users/user/Downloads/AnomaData.xlsx')
# Split the data into features (X) and target variable (y)
X = data.drop(['time', 'y'], axis=1)  # Let 'time' is not a feature for prediction
y = data['y']
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define and train the logistic regression model
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train_scaled, y_train)
y_pred = logistic_regression.predict(X_test_scaled)
y_pred
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
#What I did here..
#1. I load the data and split it into features (X) and the target variable (y).
#2.I split the data into training and testing sets using train_test_split.
#3. I standardize the features using StandardScaler.
#4. I define and train a logistic regression model with a maximum of 1000 iterations.
#5. I make predictions on the test set and evaluate the model using various evaluation metrics including accuracy, precision, recall, F1 score, and ROC AUC score.
#6. Finally, I print out the evaluation metrics to assess the performance of the logistic regression model. Adjust the code as needed based on your specific requirements and preferences.
# Makeing predictions on the test set
y_test_pred = logistic_regression.predict(X_test_scaled)
# Evaluateing the model on the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred)
print("\nTest Metrics:")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
print(f"ROC AUC Score: {roc_auc_test:.4f}")
#I make predictions on the validation set and evaluate the model using various evaluation metrics.
#I make predictions on the test set and evaluate the model using the same evaluation metrics.
#Finally, I print out the evaluation metrics for both the validation and test sets to assess the model's performance on unseen data. Adjust the code as needed based on your specific requirements and preferences.
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
# Load the data
data = pd.read_excel('C:/Users/user/Downloads/AnomaData.xlsx')
# Split the data into features (X) and target variable (y)
X = data.drop(['time', 'y'], axis=1)  # Assuming 'time' is not a feature for prediction
y = data['y']
# Train a logistic regression model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_scaled, y)
# Serialize the trained model
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(logistic_regression, f)
    # Initialize Flask application
app = Flask(__name__)
# Load the trained model
# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
    from flask import send_file

@app.route('/predict_new_one', methods=['POST'])
def predict_new_one():
    # Get input data from request
    data = request.json
    
    # Preprocess input data (if needed)
    # For example, convert input data to numpy array
    features = np.array(data['features'])
    
    # Standardize the features
    features_scaled = scaler.transform(features)
    
    # Make predictions using the loaded model
    predictions = model.predict(features_scaled)
    
    # Format predictions as JSON response
    response = {'predictions': predictions.tolist()}
    
    # Create a temporary file to store the JSON data
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(temp_file.name, 'w') as file:
        json.dump(response, file)
    
    # Close the temporary file
    temp_file.close()
    
    # Return the file for download
    return send_file(temp_file.name, as_attachment=True, attachment_filename='predictions.json')
