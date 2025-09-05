# Housing Price Prediction Project
# This script implements a machine learning pipeline to predict housing prices using the California Housing dataset.
# The pipeline includes data loading, stratified train-test splitting, preprocessing (imputation, feature engineering, 
# outlier removal, categorical encoding, and skewness reduction), model training with RandomForestRegressor, 
# hyperparameter tuning, and evaluation with RMSE and R² metrics.
# Dependencies: pandas, numpy, scikit-learn, matplotlib, zlib
# Dataset: housing.csv (expected to contain columns like longitude, latitude, median_income, ocean_proximity, etc.)

import zlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import randint

#--------------------------------------------------------------------------------------------------------------------------#
# Load Data
#--------------------------------------------------------------------------------------------------------------------------#
# Load the California Housing dataset from a CSV file.
# The dataset contains features like longitude, latitude, median_income, total_rooms, and the target variable median_house_value.
housing = pd.read_csv("housing.csv")

#--------------------------------------------------------------------------------------------------------------------------#
# Primary Analysis (Exploratory Data Analysis)
#--------------------------------------------------------------------------------------------------------------------------#
# The following commented code is used for initial data exploration to understand the dataset's structure and distributions.
# It includes displaying dataset info, statistical summary, first 10 rows, and histograms of numerical features.

# print(housing.info())  # Display dataset info (columns, types, null counts)
# print(housing.describe())  # Show statistical summary of numerical columns
# print(housing.head(10))  # Display first 10 rows
# housing.hist(bins=50, figsize=(10,9))  # Plot histograms for all numerical columns
# plt.show()

#--------------------------------------------------------------------------------------------------------------------------#
# Data Partitioning with Hash-based Splitting
#--------------------------------------------------------------------------------------------------------------------------#
# Implement a 32-bit hash-based partitioning to ensure scalable and consistent train-test splits.
# This method prevents conflicts with future data additions by using a deterministic hash function.
def is_in_test_set(identifier, test_ratio):
    """Check if an identifier belongs to the test set based on a 32-bit hash function."""
    return zlib.crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id(data, test_ratio, id_column):
    """Split data into train and test sets using a hash-based identifier."""
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_in_test_set(id_, test_ratio))
    return data[~in_test_set], data[in_test_set]

# Example usage of hash-based splitting (commented out as we use stratified splitting below):
# housing_with_id = housing.reset_index()
# train_set, test_set = split_data_with_id(housing_with_id, 0.2, "index")
# housing["id"] = housing["longitude"] * 100 + housing["latitude"]
# train_set, test_set = split_data_with_id(housing, 0.2, "id")

#--------------------------------------------------------------------------------------------------------------------------#
# Stratified Train-Test Split
#--------------------------------------------------------------------------------------------------------------------------#
# Split the dataset into train and test sets based on median_income, which has the highest Pearson correlation with the target.
# Stratified sampling ensures balanced distribution of income categories in both sets.
# Create income categories for stratification.
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

# Option 1: StratifiedShuffleSplit (commented out, used for experimentation)
# splitter = StratifiedShuffleSplit(n_splits=10, random_state=42, test_size=0.2)
# strat_splits = []
# for train_index, test_index in splitter.split(housing, housing["income_cat"]):
#     strat_train_set_n = housing.iloc[train_index]
#     strat_test_set_n = housing.iloc[test_index]
#     strat_splits.append([strat_train_set_n, strat_test_set_n])
# train_set, test_set = strat_splits[0]

# Option 2: Use train_test_split for simplicity with stratification.
train_set, test_set = train_test_split(housing, stratify=housing["income_cat"], random_state=42, test_size=0.2)

# Remove income_cat column as it was only used for stratification.
for dataset in (train_set, test_set):
    dataset.drop("income_cat", axis=1, inplace=True)

#--------------------------------------------------------------------------------------------------------------------------#
# Visualize Data on Map (Optional)
#--------------------------------------------------------------------------------------------------------------------------#
# The following commented code visualizes housing data on a map of California.
# It plots a scatter plot of longitude and latitude, with point sizes proportional to population and colors indicating median_house_value.
# housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,8), grid=True, cmap="jet", colorbar=True, 
#              s=housing["population"]/100, c="median_house_value", sharex=False, legend=True, label="population")
# filename = "california.png"
# image_city = plt.imread(filename)
# axis = -124.55, -113.95, 32.45, 42.05
# plt.axis(axis)
# plt.imshow(image_city, extent=axis)
# plt.savefig("combine_scatter_and_image.png")
# plt.show()

#--------------------------------------------------------------------------------------------------------------------------#
# Data Preprocessing: Imputation of Missing Values
#--------------------------------------------------------------------------------------------------------------------------#
# Separate features and target variable for preprocessing.
housing_features = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"]

# Check for missing values (for debugging, commented out in production).
# null_rows = housing_features.isnull().any(axis=1)
# print(housing_features[null_rows])

# Two imputation strategies are implemented:
# - IterativeImputerFeature: Uses Multivariate Imputation by Chained Equations (MICE) to predict missing values based on other features.
#   Commented out as it is computationally intensive, but included to demonstrate advanced imputation techniques.
# - SimpleImputerFeature: Uses median strategy for imputing missing values in 'total_bedrooms'. Chosen for simplicity and speed,
#   as it provided better performance in cross-validation (lower RMSE) for this dataset.

class IterativeImputerFeature(BaseEstimator, TransformerMixin):
    """Impute missing values in numerical features using IterativeImputer (MICE)."""
    def __init__(self):
        self.model = IterativeImputer(random_state=42)
    
    def fit(self, X, y=None):
        X = X.copy()
        numeric = X.select_dtypes(include=[np.number])
        self.model.fit(numeric)
        return self
    
    def transform(self, X):
        X = X.copy()
        numeric = X.select_dtypes(include=[np.number])
        imputed_numeric = pd.DataFrame(self.model.transform(numeric), index=X.index, columns=numeric.columns)
        imputed_numeric["ocean_proximity"] = X["ocean_proximity"]
        return imputed_numeric

class SimpleImputerFeature(BaseEstimator, TransformerMixin):
    """Impute missing values in 'total_bedrooms' using the median strategy."""
    def __init__(self):
        self.model = SimpleImputer(strategy="median")
    
    def fit(self, X, y=None):
        X = X.copy()
        self.model.fit(X[["total_bedrooms"]])
        return self
    
    def transform(self, X):
        X = X.copy()
        X["total_bedrooms"] = self.model.transform(X[["total_bedrooms"]])
        return X

#--------------------------------------------------------------------------------------------------------------------------#
# Feature Engineering
#--------------------------------------------------------------------------------------------------------------------------#
# Add derived features to capture relationships between existing features, improving model performance.
class AddFeatures(BaseEstimator, TransformerMixin):
    """Add derived features: rooms per bedroom, households per population, and rooms per household."""
    def __init__(self):
        self.columns = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["rooms_per_bedroom"] = X["total_rooms"] / X["total_bedrooms"]
        X["households_per_population"] = X["households"] / X["population"]
        X["rooms_per_household"] = X["total_rooms"] / X["households"]
        return X

#--------------------------------------------------------------------------------------------------------------------------#
# Outlier Removal
#--------------------------------------------------------------------------------------------------------------------------#
# Remove outliers from numerical features using Isolation Forest to improve model robustness.
class OutlierRemover(BaseEstimator, TransformerMixin):
    """Remove outliers from numerical features using Isolation Forest with 150 estimators."""
    def __init__(self):
        self.model = IsolationForest(n_jobs=4, random_state=42, n_estimators=150)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        numeric_data = X.select_dtypes([np.number])
        outlier_mask = self.model.fit_predict(numeric_data)
        return outlier_mask

#--------------------------------------------------------------------------------------------------------------------------#
# Categorical Encoding
#--------------------------------------------------------------------------------------------------------------------------#
# Convert the categorical feature 'ocean_proximity' into numerical values using one-hot encoding.
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Convert categorical 'ocean_proximity' feature into one-hot encoded columns."""
    def __init__(self):
        self.model = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    
    def fit(self, X, y=None):
        X = X.copy()
        self.model.fit(X[["ocean_proximity"]])
        return self
    
    def transform(self, X):
        X = X.copy()
        category = self.model.transform(X[["ocean_proximity"]])
        encoded_df = pd.DataFrame(category, columns=self.model.get_feature_names_out(), index=X.index)
        X = pd.concat([X, encoded_df], axis=1)
        X.drop("ocean_proximity", axis=1, inplace=True)
        return X

#--------------------------------------------------------------------------------------------------------------------------#
# Skewness Reduction and Standardization
#--------------------------------------------------------------------------------------------------------------------------#
# Apply Yeo-Johnson transformation to reduce skewness and standardize specified numerical features.
class SkewnessReducer(BaseEstimator, TransformerMixin):
    """Apply Yeo-Johnson transformation to reduce skewness in specified numerical features."""
    def __init__(self):
        self.model = PowerTransformer(method="yeo-johnson", standardize=True)
        self.columns = ["population", "total_rooms", "total_bedrooms", "households", 
                        "median_income", "rooms_per_bedroom", "rooms_per_household", 
                        "households_per_population"]
    
    def fit(self, X, y=None):
        X = X.copy()
        self.model.fit(X[self.columns])
        return self
    
    def transform(self, X):
        X = X.copy()
        transformed_df = pd.DataFrame(self.model.transform(X[self.columns]), 
                                    columns=self.model.get_feature_names_out(), index=X.index)
        X[self.columns] = transformed_df
        return X

#--------------------------------------------------------------------------------------------------------------------------#
# Geographic Feature Transformation
#--------------------------------------------------------------------------------------------------------------------------#
# Transform longitude and latitude into RBF similarity features based on KMeans clustering to capture spatial patterns.
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """Convert longitude and latitude into RBF similarity features using KMeans clustering."""
    def __init__(self, random_state=42, n_clusters=48, gamma=0.1):
        self.gamma = gamma
        self.random_state = random_state
        self.n_clusters = n_clusters
    
    def fit(self, X, y=None, sample_weight=None):
        X = X.copy()
        coords = X[["longitude", "latitude"]]
        self.model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self.model.fit(coords, sample_weight=sample_weight)
        return self
    
    def transform(self, X):
        X = X.copy()
        coords = X[["longitude", "latitude"]]
        rbf_features = rbf_kernel(coords, self.model.cluster_centers_, gamma=self.gamma)
        X["longitude"] = rbf_features[:, 0]
        X["latitude"] = rbf_features[:, 1]
        return X

#--------------------------------------------------------------------------------------------------------------------------#
# RBF Kernel Transformation
#--------------------------------------------------------------------------------------------------------------------------#
# Apply RBF kernel to housing_median_age to capture non-linear relationships.
class RBFKernelTransformer(BaseEstimator, TransformerMixin):
    """Apply RBF kernel to housing_median_age with a reference point of 35 and gamma=0.1."""
    def __init__(self):
        self.model = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35]], gamma=0.1))
    
    def fit(self, X, y=None):
        X = X.copy()
        self.model.fit(X[["housing_median_age"]])
        return self
    
    def transform(self, X):
        X = X.copy()
        rbf_features = self.model.transform(X[["housing_median_age"]])
        X["housing_median_age"] = rbf_features
        return X

#--------------------------------------------------------------------------------------------------------------------------#
# Preprocessing Pipeline
#--------------------------------------------------------------------------------------------------------------------------#
# Initial preprocessing pipeline for imputation and feature engineering.
# Uses SimpleImputerFeature for speed, but IterativeImputerFeature is available as an alternative (uncomment to test).
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputerFeature()),  # Chosen for simplicity and better cross-validation performance
    # ('imputer', IterativeImputerFeature()),  # Alternative: Advanced imputation using MICE
    ('feature_adder', AddFeatures()),
])

# Apply initial preprocessing.
preprocessed_housing = preprocessing_pipeline.fit_transform(housing_features)

# Remove outliers using Isolation Forest.
outlier_remover = OutlierRemover()
outlier_mask = outlier_remover.fit_transform(preprocessed_housing)
preprocessed_housing = preprocessed_housing[outlier_mask == 1]
housing_labels = housing_labels[outlier_mask == 1]

# Main data transformation pipeline for encoding, skewness reduction, and feature transformation.
data_pipeline = Pipeline([
    ('categorical_encoder', CategoricalEncoder()),
    ('skewness_reducer', SkewnessReducer()),
    ('geo_clustering', ClusterSimilarity(n_clusters=48)),
    ('rbf_kernel', RBFKernelTransformer()),
])

#--------------------------------------------------------------------------------------------------------------------------#
# Model Training and Evaluation
#--------------------------------------------------------------------------------------------------------------------------#
# Test multiple models to select the best one (commented out after experimentation).
# Linear Regression (tested but not used due to lower performance).

# lin_reg = LinearRegression(n_jobs=4)
# lin_reg.fit(preprocessed_housing, housing_labels)
# predictions = lin_reg.predict(preprocessed_housing)
# RMSE = root_mean_squared_error(housing_labels, predictions)
# lin_reg_rmses = -cross_val_score(lin_reg, preprocessed_housing, housing_labels, 
#                                  scoring="neg_root_mean_squared_error", cv=5, n_jobs=4)
# print(RMSE)  # Output: 66682.57
# cv_results_df = pd.Series(lin_reg_rmses)
# print(cv_results_df.describe())  # Output: mean: 66776.43, std: 1193.13




# Decision Tree Regressor (tested but prone to overfitting).

# decision_tree = DecisionTreeRegressor()
# decision_tree.fit(preprocessed_housing, housing_labels)
# predictions = decision_tree.predict(preprocessed_housing)
# RMSE = root_mean_squared_error(housing_labels, predictions)
# decision_tree_rmses = -cross_val_score(decision_tree, preprocessed_housing, housing_labels, 
#                                        scoring="neg_root_mean_squared_error", cv=5, n_jobs=4)
# print(RMSE)  # Output: 0.0 (indicates overfitting)
# cv_results_df = pd.Series(decision_tree_rmses)
# print(cv_results_df.describe())  # Output: mean: 71329.71, std: 1076.34




# Random Forest Regressor (tested and selected for final model due to better performance).

# random_forest = RandomForestRegressor(n_jobs=4, random_state=42)
# random_forest.fit(preprocessed_housing, housing_labels)
# predictions = random_forest.predict(preprocessed_housing)
# RMSE = root_mean_squared_error(housing_labels, predictions)
# random_forest_rmses = -cross_val_score(random_forest, preprocessed_housing, housing_labels, 
#                                        scoring="neg_root_mean_squared_error", cv=5, n_jobs=4)
# print(RMSE)  # Output: 18490.58
# cv_results_df = pd.Series(random_forest_rmses)
# print(cv_results_df.describe())  # Output: mean: 50274.12, std: 681.05

#--------------------------------------------------------------------------------------------------------------------------#
# Hyperparameter Tuning
#--------------------------------------------------------------------------------------------------------------------------#
# Full pipeline combining preprocessing and RandomForestRegressor with hyperparameter tuning.
full_pipeline = Pipeline([
    ("data_pipeline", data_pipeline),
    ("random_forest", RandomForestRegressor(n_jobs=4, random_state=42, max_features=8))
])

# Define parameter grids for GridSearchCV (tested but not used in final model).
param_grids = [
    {'data_pipeline__geo_clustering__n_clusters': [4, 7, 9],
     'random_forest__max_features': [4, 6, 8]},
    {'data_pipeline__geo_clustering__n_clusters': [11, 16],
     'random_forest__max_features': [6, 8, 10]},
]

grid_search = GridSearchCV(full_pipeline, param_grid=param_grids, scoring="neg_root_mean_squared_error", cv=3)

# Define parameter distributions for RandomizedSearchCV (used for final model).
param_distribs = {
    'data_pipeline__geo_clustering__n_clusters': randint(low=3, high=50),
    'random_forest__max_features': randint(low=2, high=20)
}

# Perform hyperparameter tuning with RandomizedSearchCV.
rnd_search = RandomizedSearchCV(
    full_pipeline, 
    param_distributions=param_distribs, 
    cv=4, 
    scoring="neg_root_mean_squared_error", 
    n_jobs=3, 
    n_iter=10, 
    random_state=42
)

# Fit the model with preprocessed data.
rnd_search.fit(preprocessed_housing, housing_labels)

# Extract and display cross-validation results.
cv_results_df = pd.DataFrame(rnd_search.cv_results_)
cv_results_df.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_results_df = cv_results_df[[
    "param_data_pipeline__geo_clustering__n_clusters",
    "param_random_forest__max_features",
    "split0_test_score", "split1_test_score", "split2_test_score", "mean_test_score"
]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_results_df.columns = ["n_clusters", "max_features"] + score_cols
cv_results_df[score_cols] = -cv_results_df[score_cols].round().astype(np.int64)

# print(cv_results_df.head())  # Display top hyperparameter combinations

# Get the best model and its feature importances.
final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_

# Make predictions with the final model.
predictions = final_model.predict(preprocessed_housing)

#--------------------------------------------------------------------------------------------------------------------------#
# Model Evaluation
#--------------------------------------------------------------------------------------------------------------------------#
# Evaluate the final model using RMSE and R² score.
rmse = root_mean_squared_error(predictions, housing_labels)
print(f"RMSE: {rmse}")  # Output: [depends on execution, e.g., ~18000]

r2 = r2_score(y_true=housing_labels, y_pred=predictions)
print(f"R² Score (scaled by 10): {r2 * 10}")  # Output: [depends on execution, e.g., ~9.7]

#--------------------------------------------------------------------------------------------------------------------------#
# Visualization of Results
#--------------------------------------------------------------------------------------------------------------------------#
# Plot histogram of prediction errors to analyze model performance.
errors = predictions - housing_labels
errors.hist(bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.grid(True)
plt.show()

# Scatter plot of actual vs predicted values (commented out for production).
# plt.scatter(housing_labels, predictions, alpha=0.7, color='dodgerblue')
# plt.plot([housing_labels.min(), housing_labels.max()], 
#          [predictions.min(), predictions.max()], 
#          color='red', linestyle='--')
# plt.xlabel("Actual House Value")
# plt.ylabel("Predicted House Value")
# plt.savefig("actual_vs_predicted.png")

# plt.show()
