# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load the dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

# Quick look at the data
X.head()

# Check for null values
X.isnull().sum()

# Class distribution
sns.countplot(x=y)
plt.title("Target Class Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# 2.Data Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 3. Model Training and Evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Store results
results = pd.DataFrame(columns=['Model', 'Accuracy'])

# Confusion matrix plot helper
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {title}')
    plt.show()

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression\n", classification_report(y_test, y_pred_lr))
plot_conf_matrix(y_test, y_pred_lr, "Logistic Regression")
results.loc[len(results)] = ['Logistic Regression', accuracy_score(y_test, y_pred_lr)]

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree\n", classification_report(y_test, y_pred_dt))
plot_conf_matrix(y_test, y_pred_dt, "Decision Tree")
results.loc[len(results)] = ['Decision Tree', accuracy_score(y_test, y_pred_dt)]

# Random forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest\n", classification_report(y_test, y_pred_rf))
plot_conf_matrix(y_test, y_pred_rf, "Random Forest")
results.loc[len(results)] = ['Random Forest', accuracy_score(y_test, y_pred_rf)]

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN\n", classification_report(y_test, y_pred_knn))
plot_conf_matrix(y_test, y_pred_knn, "KNN")
results.loc[len(results)] = ['KNN', accuracy_score(y_test, y_pred_knn)]

# Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Naive Bayes\n", classification_report(y_test, y_pred_nb))
plot_conf_matrix(y_test, y_pred_nb, "Naive Bayes")
results.loc[len(results)] = ['Naive Bayes', accuracy_score(y_test, y_pred_nb)]

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM\n", classification_report(y_test, y_pred_svm))
plot_conf_matrix(y_test, y_pred_svm, "SVM")
results.loc[len(results)] = ['SVM', accuracy_score(y_test, y_pred_svm)]

# 4. Results Summary
results.sort_values(by='Accuracy', ascending=False, inplace=True)
sns.barplot(data=results, x='Accuracy', y='Model')
plt.title("Model Comparison - Accuracy")
plt.xlim(0.8, 1.0)
plt.show()

results
# Save results to CSV
results.to_csv('classification_results.csv', index=False)