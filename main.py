# ============================================
# Student Performance Predictor
# Complete ML Project - Kaggle Dataset
# ============================================

# Step 1: Libraries import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, classification_report)

# ============================================
# Step 2: Dataset Load karna
# ============================================
df = pd.read_csv('student_performance.csv')

# Sirf useful columns rakhna + rename karna
df = df[['weekly_self_study_hours', 'attendance_percentage',
         'class_participation', 'total_score']].copy()

df.rename(columns={
    'weekly_self_study_hours': 'study_hours',
    'attendance_percentage':   'attendance_percent',
    'class_participation':     'internal_marks',
    'total_score':             'exam_score'
}, inplace=True)

# Pass/Fail column banana
df['pass_fail'] = (df['exam_score'] >= 40).astype(int)

# 10 lakh rows h - sample lete h 10000 ka (fast chalega)
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print("\nPehli 5 rows:")
print(df.head())
print("\nDataset size:", df.shape)
print("\nBasic Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# ============================================
# Step 3: EDA - Exploratory Data Analysis
# ============================================
print("\n" + "=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Exam Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['exam_score'], bins=30, kde=True, color='blue')
plt.title('Exam Score Distribution')
plt.xlabel('Exam Score')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Scatter Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.scatterplot(data=df, x='study_hours', y='exam_score', ax=axes[0,0], color='blue')
axes[0,0].set_title('Study Hours vs Exam Score')
sns.scatterplot(data=df, x='attendance_percent', y='exam_score', ax=axes[0,1], color='green')
axes[0,1].set_title('Attendance vs Exam Score')
sns.scatterplot(data=df, x='internal_marks', y='exam_score', ax=axes[1,0], color='red')
axes[1,0].set_title('Class Participation vs Exam Score')
plt.delaxes(axes[1,1])  # 4th plot empty h kyunki 3 hi features hain
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Box Plots
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
sns.boxplot(data=df, y='study_hours', ax=axes[0], color='blue')
axes[0].set_title('Study Hours')
sns.boxplot(data=df, y='attendance_percent', ax=axes[1], color='green')
axes[1].set_title('Attendance %')
sns.boxplot(data=df, y='internal_marks', ax=axes[2], color='red')
axes[2].set_title('Class Participation')
plt.tight_layout()
plt.show()

# ============================================
# Step 4: Preprocessing
# ============================================
print("\n" + "=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

X = df[['study_hours', 'attendance_percent', 'internal_marks']]
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training data size: {X_train_scaled.shape}")
print(f"Testing data size: {X_test_scaled.shape}")
print("Preprocessing done! ✅")

# ============================================
# Step 5: Feature Selection
# ============================================
print("\n" + "=" * 50)
print("FEATURE SELECTION")
print("=" * 50)

lr_base = LinearRegression()

# Forward Selection
forward_selector = SequentialFeatureSelector(
    lr_base, n_features_to_select=2, direction='forward')
forward_selector.fit(scaler.fit_transform(X), y)
print("\nForward Selection - Selected Features:")
print(X.columns[forward_selector.get_support()].tolist())

# Backward Selection
backward_selector = SequentialFeatureSelector(
    lr_base, n_features_to_select=2, direction='backward')
backward_selector.fit(scaler.fit_transform(X), y)
print("\nBackward Selection - Selected Features:")
print(X.columns[backward_selector.get_support()].tolist())

# PCA
print("\nPCA Analysis:")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaler.fit_transform(X))
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Covered: {sum(pca.explained_variance_ratio_):.2f}")

# PCA Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=df['exam_score'],
                      cmap='coolwarm', alpha=0.6)
plt.colorbar(scatter, label='Exam Score')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - Student Data Visualization')
plt.tight_layout()
plt.show()

# ============================================
# Step 6: Regression Models
# ============================================
print("\n" + "=" * 50)
print("REGRESSION MODELS")
print("=" * 50)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

print("\nLINEAR REGRESSION:")
print(f"R2 Score:  {r2_score(y_test, lr_pred):.2f}")
print(f"MSE:       {mean_squared_error(y_test, lr_pred):.2f}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print("\nRANDOM FOREST:")
print(f"R2 Score:  {r2_score(y_test, rf_pred):.2f}")
print(f"MSE:       {mean_squared_error(y_test, rf_pred):.2f}")

# Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, color='blue', alpha=0.5)
plt.plot([0, 100], [0, 100], color='red', linewidth=2)
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('Actual vs Predicted Exam Scores (Random Forest)')
plt.tight_layout()
plt.show()

# ============================================
# Step 7: Classification - Pass/Fail
# ============================================
print("\n" + "=" * 50)
print("CLASSIFICATION - PASS/FAIL PREDICTION")
print("=" * 50)

X_clf = df[['study_hours', 'attendance_percent', 'internal_marks']]
y_clf = df['pass_fail']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42)

scaler2 = StandardScaler()
X_train_c = scaler2.fit_transform(X_train_c)
X_test_c = scaler2.transform(X_test_c)

clf_model = LogisticRegression()
clf_model.fit(X_train_c, y_train_c)
clf_pred = clf_model.predict(X_test_c)

print(f"\nAccuracy:  {accuracy_score(y_test_c, clf_pred):.2f}")
print(f"Precision: {precision_score(y_test_c, clf_pred):.2f}")
print(f"Recall:    {recall_score(y_test_c, clf_pred):.2f}")
print("\nDetailed Report:")
print(classification_report(y_test_c, clf_pred))

print("\n" + "=" * 50)
print("✅ PROJECT COMPLETE!")
print("=" * 50)
