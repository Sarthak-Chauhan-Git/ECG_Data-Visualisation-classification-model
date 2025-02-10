# Install necessary libraries
!pip install numpy==1.23.5 pandas==1.5.3 sweetviz==2.1.4 lazypredict scikit-learn

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeRegressor

# Download dataset using kagglehub
import kagglehub
path = kagglehub.dataset_download("gkshitij/mit-bih-arrhythmia-database-new")
print("Path to dataset files:", path)

# Load the dataset
file_path = "/root/.cache/kagglehub/datasets/gkshitij/mit-bih-arrhythmia-database-new/versions/1/MIT-BIH Arrhythmia Database new.csv"
data = pd.read_csv(file_path)

# Data Exploration and Preprocessing
data.info()
data.describe()
numeric_data = data.select_dtypes(include=["float64", "int64"])

# Visualizations
def visualize_data(numeric_data):
    # Scatterplot for all features
    for col in numeric_data.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(numeric_data[col])), numeric_data[col], alpha=0.5)
        plt.title(f"Scatterplot for {col}")
        plt.show()

    # Boxplot for all features
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=numeric_data)
    plt.xticks(rotation=90)
    plt.title("Boxplot for Numeric Features")
    plt.tight_layout()
    plt.show()

    # Pairplot for first 5 features
    sns.pairplot(numeric_data.iloc[:, :5])
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

visualize_data(numeric_data)

# Identify Target Variable
def identify_target(data):
    candidates = data.nunique()[data.nunique() < 20].index.tolist()
    potential_targets = [col for col in candidates if data[col].dtype in ["int64", "float64", "object"]]
    print(f"Potential target columns: {potential_targets}")
    return potential_targets[0] if potential_targets else None

target_column = identify_target(data)
if not target_column:
    raise ValueError("Target variable could not be identified.")

# PCA on ECG Features
scaler = StandardScaler()
ecg_features = numeric_data
ecg_scaled = scaler.fit_transform(ecg_features)

pca = PCA()
ecg_pca = pca.fit_transform(ecg_scaled)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="--", color="b")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by Principal Components")
plt.grid()
plt.show()

n_components = next(i for i, total in enumerate(cumulative_variance) if total >= 0.95) + 1
print(f"Number of components explaining ~95% variance: {n_components}")
pca_final = PCA(n_components=n_components)
ecg_pca_final = pca_final.fit_transform(ecg_scaled)

ecg_pca_df = pd.DataFrame(ecg_pca_final, columns=[f"PC{i+1}" for i in range(n_components)])
ecg_pca_df.to_csv("ecg_pca_transformed.csv", index=False)
print("PCA-transformed ECG data saved to 'ecg_pca_transformed.csv'.")

# Lazy Regressor
data = pd.read_csv(file_path)
data = data.sample(frac=0.1, random_state=42)
target_column = data.columns[-1]
X = data.drop(target_column, axis=1)
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
print(models.sort_values(by=['R-Squared', 'RMSE'], ascending=[False, True]).head(4))

# SMOTE for Oversampling
target_column = 'type'
X = data.drop(target_column, axis=1)
y = data[target_column]
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target_column])], axis=1)
print("Original data shape:", data.shape)
print("Resampled data shape:", resampled_data.shape)

# Applying Models
target_column = 'type'
X = data.drop(target_column, axis=1)
y = data[target_column]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_classes = [round(value) for value in y_pred]
y_pred_classes = [min(max(value, 0), len(label_encoder.classes_) - 1) for value in y_pred_classes]
y_pred_classes = label_encoder.inverse_transform(y_pred_classes)
y_test_classes = label_encoder.inverse_transform(y_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"GradientBoostingRegressor - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}, Accuracy: {accuracy * 100:.2f}%")

# 2. BaggingRegressor
base_estimator = DecisionTreeRegressor(random_state=42)
model = BaggingRegressor(estimator=base_estimator, n_estimators=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. ExtraTreesRegressor
model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. HistGradientBoostingRegressor
model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. RandomForestRegressor with sampling
sample_fraction = 0.2
X_train_sample = X_train.sample(frac=sample_fraction, random_state=42)
y_train = pd.Series(y_train, index=X_train.index)
y_train_sample = y_train[X_train_sample.index.intersection(y_train.index)]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_sample, y_train_sample)
y_pred = model.predict(X_test)
