import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gdown  # To download from Google Drive
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests

# Function to download data from Dropbox
def download_from_dropbox(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

# Dropbox direct download link and output path
dropbox_file_url = "https://dl.dropboxusercontent.com/scl/fi/8ud8f7aw611auu795wtgr/FraudTransactionsLog.csv?rlkey=ztx9opt31k4hlookuzm88osi7&st=iadmhp6f&dl=1"
data_path = "FraudTransactionsLog.csv"

# Download the file from Dropbox
download_from_dropbox(dropbox_file_url, data_path)

# Load data with caching to improve performance
@st.cache_resource
def load_data(data_path, sample_fraction):
    data = pd.read_csv(data_path, sep=',')
    data_sampled = data.sample(frac=sample_fraction, random_state=42)
    return data_sampled

# Use the same data processing as before
sample_fraction = 0.05  # Adjust as needed
data = load_data(data_path, sample_fraction)

# Streamlit app continues as before...


# Your Streamlit code continues...

# Check if columns exist before dropping
columns_to_drop = ['nameOrig', 'nameDest']
existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

correlation_data = data.drop(existing_columns_to_drop, axis=1)

# Convert categorical data to dummy variables
correlation_data = pd.get_dummies(correlation_data, columns=['type'])

# Streamlit UI
st.title("Financial Fraud Detection Analysis")

# Data Information
st.header("Data Information")
st.write(data.describe())

# Null value check
st.header("Null Values")
null_values = data.isnull().sum()
st.write(null_values)

# Fraud vs Non-Fraud Transactions as Pie Chart
st.header("Fraud vs Non-Fraud Transactions")
is_fraud_count = data['isFraud'].value_counts(normalize=True) * 100
if st.button("Show Fraud vs Non-Fraud Transactions (Pie Chart)"):
    plt.figure()
    is_fraud_count.plot(kind='pie', autopct='%1.1f%%', labels=['Non-Fraud', 'Fraud'], colors=['lightblue', 'orange'])
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    st.pyplot(plt)

# Correlation Matrix
st.header("Correlation Matrix")
if st.button("Show Correlation Matrix"):
    correlation_matrix = correlation_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

# Transaction Types
st.header("Transaction Types")
transaction_type_counts = data['type'].value_counts(normalize=True)
if st.button("Show Transaction Types"):
    plt.figure(figsize=(7, 5))
    transaction_type_counts.plot(kind='bar', color=sns.color_palette("Set3"))
    st.pyplot(plt)

# Distribution of Transactions by Step
st.header("Distribution of Transactions by Step")
if st.button("Show Distribution by Step"):
    plt.figure(figsize=(7, 5))
    sns.histplot(data['step'], kde=True, color='orange')
    st.pyplot(plt)

# Fraudulent Transactions by Type
st.header("Fraudulent Transactions by Type")
fraud_percentage = (data[data['isFraud'] == 1].groupby('type').size() /
                    data[data['isFraud'] == 1].shape[0]) * 100
if st.button("Show Fraudulent Transactions by Type"):
    plt.figure(figsize=(7, 5))
    fraud_percentage.plot(kind='bar', color=sns.color_palette("Set3"))
    st.pyplot(plt)

# Model Training and Performance
data['origBalance_inacc'] = (data['oldbalanceOrg'] - data['amount']) - data['newbalanceOrig']
data['destBalance_inacc'] = (data['oldbalanceDest'] + data['amount']) - data['newbalanceDest']

# Drop unnecessary columns and create dummy variables
data.drop(existing_columns_to_drop, axis=1, inplace=True)
data = pd.get_dummies(data, columns=['type'])

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('isFraud', axis=1))

# Split the data
X = pd.DataFrame(scaled_features, columns=data.drop('isFraud', axis=1).columns)
y = data['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate models
performance = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    confusion_mat = confusion_matrix(y_test, y_pred)

    performance[name] = {
        "Accuracy": accuracy,
        "Classification Report": classification_rep,
        "Confusion Matrix": confusion_mat
    }

# Model Performance Comparison
st.header("Model Performance Comparison")

# Bar plot for accuracy comparison among models
st.subheader("Model Accuracy Comparison")
model_names = list(performance.keys())
accuracies = [performance[name]['Accuracy'] for name in model_names]
plt.figure()
plt.bar(model_names, accuracies, color='skyblue')
plt.xlabel("Model")
plt.ylabel("Accuracy")
st.pyplot(plt)

# Confusion Matrix Heatmap
st.subheader("Confusion Matrices")
for model_name, results in performance.items():
    if st.button(f"Show Confusion Matrix for {model_name}"):
        plt.figure(figsize=(6, 6))
        sns.heatmap(results["Confusion Matrix"], annot=True, fmt="d", cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

# Instructions to run the Streamlit app
st.header("Running the Streamlit App")
st.write("To run this Streamlit app, execute the following command in your terminal:")
st.code("streamlit run streamlit_app.py")

# streamlit_app.py


# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Cache data loading to improve performance
# data_path = 'FraudTransactionsLog.csv'

# @st.cache_resource
# def load_data(data_path, sample_fraction):
#     # Load and sample data
#     data = pd.read_csv(data_path, sep=',')
#     data_sampled = data.sample(frac=sample_fraction, random_state=42)
#     return data_sampled

# # Load a smaller fraction of the data to improve speed
# sample_fraction = 0.05  # Adjust as needed to control data size
# data = load_data(data_path, sample_fraction)

# # Check if columns exist before dropping
# columns_to_drop = ['nameOrig', 'nameDest']
# existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

# correlation_data = data.drop(existing_columns_to_drop, axis=1)

# # Convert categorical data to dummy variables
# correlation_data = pd.get_dummies(correlation_data, columns=['type'])

# # Streamlit UI
# st.title("Financial Fraud Detection Analysis")

# # Data Information
# st.header("Data Information")
# st.write(data.describe())

# # Null value check
# st.header("Null Values")
# null_values = data.isnull().sum()
# st.write(null_values)

# # Fraud vs Non-Fraud Transactions as Pie Chart
# st.header("Fraud vs Non-Fraud Transactions")
# is_fraud_count = data['isFraud'].value_counts(normalize=True) * 100
# if st.button("Show Fraud vs Non-Fraud Transactions (Pie Chart)"):
#     plt.figure()
#     is_fraud_count.plot(kind='pie', autopct='%1.1f%%', labels=['Non-Fraud', 'Fraud'], colors=['lightblue', 'orange'])
#     plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
#     st.pyplot(plt)

# # Correlation Matrix
# st.header("Correlation Matrix")
# if st.button("Show Correlation Matrix"):
#     correlation_matrix = correlation_data.corr()
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#     st.pyplot(plt)

# # Transaction Types
# st.header("Transaction Types")
# transaction_type_counts = data['type'].value_counts(normalize=True)
# if st.button("Show Transaction Types"):
#     plt.figure(figsize=(7, 5))
#     transaction_type_counts.plot(kind='bar', color=sns.color_palette("Set3"))
#     st.pyplot(plt)

# # Distribution of Transactions by Step
# st.header("Distribution of Transactions by Step")
# if st.button("Show Distribution by Step"):
#     plt.figure(figsize=(7, 5))
#     sns.histplot(data['step'], kde=True, color='orange')
#     st.pyplot(plt)

# # Fraudulent Transactions by Type
# st.header("Fraudulent Transactions by Type")
# fraud_percentage = (data[data['isFraud'] == 1].groupby('type').size() /
#                     data[data['isFraud'] == 1].shape[0]) * 100
# if st.button("Show Fraudulent Transactions by Type"):
#     plt.figure(figsize=(7, 5))
#     fraud_percentage.plot(kind='bar', color=sns.color_palette("Set3"))
#     st.pyplot(plt)

# # Model Training and Performance
# data['origBalance_inacc'] = (data['oldbalanceOrg'] - data['amount']) - data['newbalanceOrig']
# data['destBalance_inacc'] = (data['oldbalanceDest'] + data['amount']) - data['newbalanceDest']

# # Drop unnecessary columns and create dummy variables
# data.drop(existing_columns_to_drop, axis=1, inplace=True)
# data = pd.get_dummies(data, columns=['type'])

# # Scale the data
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(data.drop('isFraud', axis=1))

# # Split the data
# X = pd.DataFrame(scaled_features, columns=data.drop('isFraud', axis=1).columns)
# y = data['isFraud']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # Initialize models
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier()
# }

# # Train and evaluate models
# performance = {}
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     classification_rep = classification_report(y_test, y_pred, output_dict=True)
#     confusion_mat = confusion_matrix(y_test, y_pred)

#     performance[name] = {
#         "Accuracy": accuracy,
#         "Classification Report": classification_rep,
#         "Confusion Matrix": confusion_mat
#     }

# # Model Performance Comparison
# st.header("Model Performance Comparison")

# # Bar plot for accuracy comparison among models
# st.subheader("Model Accuracy Comparison")
# model_names = list(performance.keys())
# accuracies = [performance[name]['Accuracy'] for name in model_names]
# plt.figure()
# plt.bar(model_names, accuracies, color='skyblue')
# plt.xlabel("Model")
# plt.ylabel("Accuracy")
# st.pyplot(plt)

# # Confusion Matrix Heatmap
# st.subheader("Confusion Matrices")
# for model_name, results in performance.items():
#     if st.button(f"Show Confusion Matrix for {model_name}"):
#         plt.figure(figsize=(6, 6))
#         sns.heatmap(results["Confusion Matrix"], annot=True, fmt="d", cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         st.pyplot(plt)

# # Instructions to run the Streamlit app
# st.header("Running the Streamlit App")
# st.write("To run this Streamlit app, execute the following command in your terminal:")
# st.code("streamlit run streamlit_app.py")

# # streamlit_app.py

# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the data
# data_path = 'FraudTransactionsLog.csv'
# data = pd.read_csv(data_path, sep=',')

# correlation_data = data.drop(['nameOrig', 'nameDest'], axis=1)

# # Create dummy variables for categorical data
# correlation_data = pd.get_dummies(correlation_data, columns=['type'])

# # Correlation Matrix
# st.title("Financial Fraud Detection Analysis")

# st.header("Correlation Matrix")
# correlation_matrix = correlation_data.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# st.pyplot(plt)

# # Basic data information
# st.header("Data Information")
# st.write(data.info())
# st.write(data.describe())

# # Null value check
# st.header("Null Values")
# null_values = data.isnull().sum()
# st.write(null_values)

# # Fraud percentage
# st.header("Fraud vs Non-Fraud Transactions")
# is_fraud_count = data['isFraud'].value_counts(normalize=True) * 100
# is_fraud_count.plot(kind='bar', color=['lightblue', 'orange'], figsize=(7, 5))
# st.pyplot(plt)

# # Transaction types
# st.header("Transaction Types")
# transaction_type_counts = data['type'].value_counts(normalize=True)
# transaction_type_counts.plot(kind='bar', color=sns.color_palette("Set3"), figsize=(7, 5))
# st.pyplot(plt)

# # Transaction amount histogram
# st.header("Transaction Amount Distribution")
# plt.figure(figsize=(7, 5))
# sns.histplot(data['amount'], kde=True, color='skyblue')
# st.pyplot(plt)

# # Fraudulent transaction types
# st.header("Fraudulent Transactions by Type")
# fraud_percentage = (data[data['isFraud'] == 1].groupby('type').size() /
#                     data[data['isFraud'] == 1].shape[0]) * 100
# fraud_percentage.plot(kind='bar', color=sns.color_palette("Set3"), figsize=(7, 5))
# st.pyplot(plt)

# # Model Training and Performance
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Data preparation
# data['origBalance_inacc'] = (data['oldbalanceOrg'] - data['amount']) - data['newbalanceOrig']
# data['destBalance_inacc'] = (data['oldbalanceDest'] + data['amount']) - data['newbalanceDest']

# # Drop unnecessary columns and create dummy variables
# data.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)
# data = pd.get_dummies(data, columns=['type'])

# # Scale the data
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(data.drop('isFraud', axis=1))

# # Split the data
# X = pd.DataFrame(scaled_features, columns=data.drop('isFraud', axis=1).columns)
# y = data['isFraud']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # Initialize models
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier()
# }

# # Train and evaluate models
# performance = {}
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     classification_rep = classification_report(y_test, y_pred, output_dict=True)
#     confusion_mat = confusion_matrix(y_test, y_pred)
    
#     performance[name] = {
#         "Accuracy": accuracy,
#         "Classification Report": classification_rep,
#         "Confusion Matrix": confusion_mat
#     }

# # Model Performance Comparison
# st.header("Model Performance Comparison")

# for model_name, results in performance.items():
#     st.subheader(model_name)
#     st.write(f"Accuracy: {results['Accuracy']:.2f}")
    
#     # Display classification report as a DataFrame
#     classification_df = pd.DataFrame(results["Classification Report"]).transpose()
#     st.write("Classification Report:", classification_df)
    
#     # Confusion Matrix
#     plt.figure(figsize=(6, 6))
#     sns.heatmap(results["Confusion Matrix"], annot=True, fmt="d", cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     st.pyplot(plt)

# # How to run this Streamlit app
# st.header("Running the Streamlit App")
# st.write("To run this Streamlit app, execute the following command in your terminal:")
# st.code("streamlit run streamlit_app.py")

# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Load the data with caching
# data_path = 'FraudTransactionsLog.csv'

# @st.cache_resource
# def load_data(data_path, sample_fraction):
#     # Load the data and sample a subset
#     data = pd.read_csv(data_path, sep=',')
#     data_sampled = data.sample(frac=sample_fraction, random_state=42)
#     return data_sampled

# # Load and preprocess data
# sample_fraction = 0.1
# data = load_data(data_path, sample_fraction)

# # Check if columns exist before dropping
# columns_to_drop = ['nameOrig', 'nameDest']
# existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

# correlation_data = data.drop(existing_columns_to_drop, axis=1)

# # Convert categorical data to dummy variables
# correlation_data = pd.get_dummies(correlation_data, columns=['type'])

# # Streamlit UI
# st.title("Financial Fraud Detection Analysis")

# # Data Information
# st.header("Data Information")
# st.write(data.info())
# st.write(data.describe())

# # Null value check
# st.header("Null Values")
# null_values = data.isnull().sum()
# st.write(null_values)

# # Fraud vs Non-Fraud Transactions
# st.header("Fraud vs Non-Fraud Transactions")
# is_fraud_count = data['isFraud'].value_counts(normalize=True) * 100
# if st.button("Show Fraud vs Non-Fraud Transactions"):
#     plt.figure(figsize=(7, 5))
#     is_fraud_count.plot(kind='bar', color=['lightblue', 'orange'])
#     st.pyplot(plt)

# # Correlation Matrix
# st.header("Correlation Matrix")
# if st.button("Show Correlation Matrix"):
#     correlation_matrix = correlation_data.corr()
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#     st.pyplot(plt)

# # Transaction Types
# st.header("Transaction Types")
# transaction_type_counts = data['type'].value_counts(normalize=True)
# if st.button("Show Transaction Types"):
#     plt.figure(figsize=(7, 5))
#     transaction_type_counts.plot(kind='bar', color=sns.color_palette("Set3"))
#     st.pyplot(plt)

# # Transaction Amount Distribution
# st.header("Transaction Amount Distribution")
# if st.button("Show Transaction Amount Distribution"):
#     plt.figure(figsize=(7, 5))
#     sns.histplot(data['amount'], kde=True, color='skyblue')
#     st.pyplot(plt)

# # Fraudulent Transactions by Type
# st.header("Fraudulent Transactions by Type")
# fraud_percentage = (data[data['isFraud'] == 1].groupby('type').size() /
#                     data[data['isFraud'] == 1].shape[0]) * 100
# if st.button("Show Fraudulent Transactions by Type"):
#     plt.figure(figsize=(7, 5))
#     fraud_percentage.plot(kind='bar', color=sns.color_palette("Set3"))
#     st.pyplot(plt)

# # Model Training and Performance
# data['origBalance_inacc'] = (data['oldbalanceOrg'] - data['amount']) - data['newbalanceOrig']
# data['destBalance_inacc'] = (data['oldbalanceDest'] + data['amount']) - data['newbalanceDest']

# # Drop unnecessary columns and create dummy variables
# data.drop(existing_columns_to_drop, axis=1, inplace=True)
# data = pd.get_dummies(data, columns=['type'])

# # Scale the data
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(data.drop('isFraud', axis=1))

# # Split the data
# X = pd.DataFrame(scaled_features, columns=data.drop('isFraud', axis=1).columns)
# y = data['isFraud']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # Initialize models
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier()
# }

# # Train and evaluate models
# performance = {}
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     classification_rep = classification_report(y_test, y_pred, output_dict=True)
#     confusion_mat = confusion_matrix(y_test, y_pred)

#     performance[name] = {
#         "Accuracy": accuracy,
#         "Classification Report": classification_rep,
#         "Confusion Matrix": confusion_mat
#     }

# # Model Performance Comparison
# st.header("Model Performance Comparison")

# for model_name, results in performance.items():
#     st.subheader(model_name)
#     st.write(f"Accuracy: {results['Accuracy']:.2f}")

#     # Display classification report as a DataFrame
#     classification_df = pd.DataFrame(results["Classification Report"]).transpose()
#     st.write("Classification Report:", classification_df)

#     # Confusion Matrix
#     if st.button(f"Show Confusion Matrix for {model_name}"):
#         plt.figure(figsize=(6, 6))
#         sns.heatmap(results["Confusion Matrix"], annot=True, fmt="d", cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         st.pyplot(plt)

# # Instructions to run the Streamlit app
# st.header("Running the Streamlit App")
# st.write("To run this Streamlit app, execute the following command in your terminal:")
# st.code("streamlit run streamlit_app.py")
