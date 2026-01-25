"""
Person 1: Model Building Pipeline
Datathon 2026 - Tasks 1.7-1.9

This script:
- Task 1.7: Prepares training data from merged dataset
- Task 1.8: Trains Random Forest model to predict funding needs
- Task 1.9: Generates predictions and identifies funding gaps
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DATATHON 2026 - MODEL BUILDING PIPELINE")
print("=" * 70)

# ============================================================================
# TASK 1.7: Prepare Training Data
# ============================================================================
print("\n" + "=" * 70)
print("TASK 1.7: Preparing Training Data")
print("=" * 70)

# Load complete dataset
df = pd.read_csv('complete_funding_dataset.csv')
print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

# Load effectiveness scores
scores = pd.read_csv('person2_effectiveness_scores.csv')

# Define feature columns
numeric_features = [
    # INFORM severity metrics
    'INFORM_Mean', 'INFORM_Std', 'INFORM_Min', 'INFORM_Max',
    'People_In_Need_Avg', 'Complexity_Avg', 'Impact_Avg',
    
    # Economic indicators
    'GDP_Per_Capita', 'Inflation_Rate',
    
    # Population metrics  
    'Population', 'Vulnerable_Pop_Pct', 'IDP_Rate',
    
    # Crisis metrics
    'Number_Clusters', 'Total_In_Need', 'Coverage_Rate',
    
    # Derived features
    'Need_Per_Capita', 'Economic_Stress',
]

categorical_features = ['Crisis_Type', 'UN_Region']

# Target variable options
target_options = {
    'FTS_Funding': 'Actual funding received (FTS)',
    'FTS_Requirements': 'Funding requirements (FTS)',
    'Total_Funding': 'Total funding (CERF+CBPF+HRP)',
    'Effectiveness_Score': 'Effectiveness score (0-100)'
}

print("\nTarget variable options:")
for key, desc in target_options.items():
    non_null = df[key].notna().sum() if key in df.columns else 0
    print(f"  - {key}: {desc} ({non_null} non-null)")

# Use FTS_Funding as primary target (most comprehensive)
TARGET = 'FTS_Funding'
print(f"\nUsing target variable: {TARGET}")

# Filter to rows with valid target
df_model = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
print(f"Rows with valid target: {len(df_model)}")

# Prepare features
print("\n--- Preparing Features ---")

# Check which numeric features are available
available_numeric = [f for f in numeric_features if f in df_model.columns]
missing_numeric = [f for f in numeric_features if f not in df_model.columns]
print(f"Available numeric features: {len(available_numeric)}")
if missing_numeric:
    print(f"Missing features: {missing_numeric}")

# Fill missing values with median
X_numeric = df_model[available_numeric].copy()
for col in available_numeric:
    median_val = X_numeric[col].median()
    X_numeric[col] = X_numeric[col].fillna(median_val)
    
print(f"Numeric features shape: {X_numeric.shape}")

# Encode categorical features
X_categorical = pd.DataFrame()
label_encoders = {}

for cat_col in categorical_features:
    if cat_col in df_model.columns:
        # One-hot encoding
        dummies = pd.get_dummies(df_model[cat_col], prefix=cat_col, drop_first=True)
        X_categorical = pd.concat([X_categorical, dummies], axis=1)
        print(f"  {cat_col}: {df_model[cat_col].nunique()} categories -> {len(dummies.columns)} dummy vars")

print(f"Categorical features shape: {X_categorical.shape}")

# Combine features
X = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
y = df_model[TARGET].reset_index(drop=True)

# Log transform target for better distribution
y_log = np.log1p(y)

print(f"\nFinal feature matrix: {X.shape}")
print(f"Target range: ${y.min():,.0f} to ${y.max():,.0f}")
print(f"Target mean: ${y.mean():,.0f}")

# ============================================================================
# TASK 1.8: Train Models
# ============================================================================
print("\n" + "=" * 70)
print("TASK 1.8: Training Models")
print("=" * 70)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}

# --- Model 1: Random Forest ---
print("\n--- Training Random Forest ---")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predictions (transform back from log)
y_pred_rf_log = rf.predict(X_test)
y_pred_rf = np.expm1(y_pred_rf_log)
y_test_actual = np.expm1(y_test)

# Metrics
rf_r2 = r2_score(y_test, y_pred_rf_log)
rf_mae = mean_absolute_error(y_test_actual, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_rf))

print(f"  R² Score: {rf_r2:.4f}")
print(f"  MAE: ${rf_mae:,.0f}")
print(f"  RMSE: ${rf_rmse:,.0f}")

# Cross-validation
cv_scores = cross_val_score(rf, X, y_log, cv=5, scoring='r2')
print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results['Random Forest'] = {
    'model': rf,
    'r2': rf_r2,
    'mae': rf_mae,
    'rmse': rf_rmse,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

# --- Model 2: Gradient Boosting ---
print("\n--- Training Gradient Boosting ---")
gb = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=5,
    random_state=42
)
gb.fit(X_train, y_train)

y_pred_gb_log = gb.predict(X_test)
y_pred_gb = np.expm1(y_pred_gb_log)

gb_r2 = r2_score(y_test, y_pred_gb_log)
gb_mae = mean_absolute_error(y_test_actual, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_gb))

print(f"  R² Score: {gb_r2:.4f}")
print(f"  MAE: ${gb_mae:,.0f}")
print(f"  RMSE: ${gb_rmse:,.0f}")

cv_scores_gb = cross_val_score(gb, X, y_log, cv=5, scoring='r2')
print(f"  CV R² Score: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std():.4f})")

results['Gradient Boosting'] = {
    'model': gb,
    'r2': gb_r2,
    'mae': gb_mae,
    'rmse': gb_rmse,
    'cv_mean': cv_scores_gb.mean(),
    'cv_std': cv_scores_gb.std()
}

# --- Model 3: Ridge Regression (baseline) ---
print("\n--- Training Ridge Regression (baseline) ---")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred_ridge_log = ridge.predict(X_test_scaled)
y_pred_ridge = np.expm1(y_pred_ridge_log)

ridge_r2 = r2_score(y_test, y_pred_ridge_log)
ridge_mae = mean_absolute_error(y_test_actual, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_ridge))

print(f"  R² Score: {ridge_r2:.4f}")
print(f"  MAE: ${ridge_mae:,.0f}")
print(f"  RMSE: ${ridge_rmse:,.0f}")

results['Ridge Regression'] = {
    'model': ridge,
    'r2': ridge_r2,
    'mae': ridge_mae,
    'rmse': ridge_rmse,
    'cv_mean': None,
    'cv_std': None
}

# Select best model
best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
best_model = results[best_model_name]['model']
print(f"\n*** Best Model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f}) ***")

# --- Feature Importance (from Random Forest) ---
print("\n--- Feature Importance (Random Forest) ---")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# ============================================================================
# TASK 1.9: Generate Predictions & Identify Gaps
# ============================================================================
print("\n" + "=" * 70)
print("TASK 1.9: Generating Predictions for All Crises")
print("=" * 70)

# Prepare full dataset for prediction
df_full = df.copy()

# Prepare features for all rows
X_full_numeric = df_full[available_numeric].copy()
for col in available_numeric:
    median_val = X_full_numeric[col].median()
    X_full_numeric[col] = X_full_numeric[col].fillna(median_val)

X_full_categorical = pd.DataFrame()
for cat_col in categorical_features:
    if cat_col in df_full.columns:
        dummies = pd.get_dummies(df_full[cat_col], prefix=cat_col, drop_first=True)
        X_full_categorical = pd.concat([X_full_categorical, dummies], axis=1)

X_full = pd.concat([X_full_numeric.reset_index(drop=True), X_full_categorical.reset_index(drop=True)], axis=1)

# Ensure columns match training data
missing_cols = set(X.columns) - set(X_full.columns)
for col in missing_cols:
    X_full[col] = 0
X_full = X_full[X.columns]

# Generate predictions using best model (Random Forest)
y_pred_full_log = rf.predict(X_full)
y_pred_full = np.expm1(y_pred_full_log)

# Add predictions to dataframe
df_full['Predicted_Funding'] = y_pred_full
df_full['Actual_Funding'] = df_full['FTS_Funding'].fillna(0)
df_full['Funding_Gap'] = df_full['Predicted_Funding'] - df_full['Actual_Funding']
df_full['Gap_Percentage'] = (
    df_full['Funding_Gap'] / df_full['Predicted_Funding'].replace(0, np.nan) * 100
).fillna(0)

# Categorize funding status
def categorize_funding(row):
    if pd.isna(row['Actual_Funding']) or row['Actual_Funding'] == 0:
        return 'No Funding Data'
    gap_pct = row['Gap_Percentage']
    if gap_pct > 50:
        return 'Severely Underfunded'
    elif gap_pct > 20:
        return 'Underfunded'
    elif gap_pct > -20:
        return 'Adequately Funded'
    else:
        return 'Well Funded'

df_full['Funding_Status'] = df_full.apply(categorize_funding, axis=1)

print("\n--- Funding Status Distribution ---")
print(df_full['Funding_Status'].value_counts().to_string())

print("\n--- Top 10 Underfunded Crises (by Gap) ---")
underfunded = df_full[
    (df_full['Actual_Funding'] > 0) & 
    (df_full['INFORM_Mean'] >= 3.0)
].nlargest(10, 'Funding_Gap')
cols_show = ['ISO3', 'Year', 'Country', 'INFORM_Mean', 'Actual_Funding', 
             'Predicted_Funding', 'Funding_Gap', 'Funding_Status']
print(underfunded[cols_show].to_string(index=False))

print("\n--- Top 10 Best Funded Crises ---")
well_funded = df_full[df_full['Actual_Funding'] > 0].nsmallest(10, 'Gap_Percentage')
print(well_funded[cols_show].to_string(index=False))

# ============================================================================
# Save Outputs
# ============================================================================
print("\n" + "=" * 70)
print("SAVING OUTPUTS")
print("=" * 70)

# Save model
with open('person1_model.pkl', 'wb') as f:
    pickle.dump({
        'model': rf,
        'scaler': scaler,
        'feature_columns': X.columns.tolist(),
        'target': TARGET
    }, f)
print("Saved: person1_model.pkl")

# Save predictions
prediction_cols = ['ISO3', 'Year', 'Country', 'INFORM_Mean', 'Crisis_Type', 'UN_Region',
                   'Actual_Funding', 'Predicted_Funding', 'Funding_Gap', 'Gap_Percentage',
                   'Funding_Status', 'Effectiveness_Score', 'Effectiveness_Category']
df_full[prediction_cols].to_csv('person1_all_predictions.csv', index=False)
print("Saved: person1_all_predictions.csv")

# Save feature importance
feature_importance.to_csv('person1_feature_importance.csv', index=False)
print("Saved: person1_feature_importance.csv")

# Save model performance
performance = pd.DataFrame([
    {'Model': name, 'R2': r['r2'], 'MAE': r['mae'], 'RMSE': r['rmse'], 
     'CV_R2_Mean': r['cv_mean'], 'CV_R2_Std': r['cv_std']}
    for name, r in results.items()
])
performance.to_csv('person1_model_performance.csv', index=False)
print("Saved: person1_model_performance.csv")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("MODEL BUILDING COMPLETE - SUMMARY")
print("=" * 70)

print(f"""
TRAINING DATA:
- Samples used: {len(X)}
- Features: {len(X.columns)}
- Target: {TARGET}

BEST MODEL: {best_model_name}
- R² Score: {results[best_model_name]['r2']:.4f}
- Mean Absolute Error: ${results[best_model_name]['mae']:,.0f}
- Cross-validation R²: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std']:.4f})

TOP PREDICTIVE FEATURES:
{feature_importance.head(5).to_string(index=False)}

PREDICTIONS GENERATED:
- Total crises analyzed: {len(df_full)}
- Severely Underfunded: {(df_full['Funding_Status'] == 'Severely Underfunded').sum()}
- Underfunded: {(df_full['Funding_Status'] == 'Underfunded').sum()}
- Adequately Funded: {(df_full['Funding_Status'] == 'Adequately Funded').sum()}
- Well Funded: {(df_full['Funding_Status'] == 'Well Funded').sum()}

FILES CREATED:
1. person1_model.pkl - Trained Random Forest model
2. person1_all_predictions.csv - Predictions for all crises
3. person1_feature_importance.csv - Feature importance rankings
4. person1_model_performance.csv - Model comparison metrics
""")
