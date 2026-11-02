"""
Data Mining & Machine Learning Pipeline for Used Car Price Prediction & Automobile Analytics
Implements:
1. Missing Value Detection & Imputation (Median / Mode)
2. Outlier Detection (IQR & Z-Score) & Treatment (Winsorization/Capping)
3. Data Cleaning (Duplicates removal, irrelevant column drops)
4. Data Transformation (StandardScaler / MinMaxScaler & One-Hot Encoding)
5. Feature Selection (Correlation Analysis & Random Forest Feature Importance)
6. Exploratory Data Analysis (Plotly Visualizations: Distributions, Box plots, Scatters, Heatmap)
7. Model Training & Evaluation (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting)
8. Serialization (joblib) & Real-time inference
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
DEFAULT_DATASET = os.path.join(DATA_DIR, 'used_cars_dataset.csv')


class CarPricePipeline:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.best_model_name = "Random Forest Regressor"
        self.best_model = None
        self.scaler = None
        self.numerical_cols = ['year', 'mileage', 'engine_size', 'horsepower', 'fuel_efficiency', 'seats']
        self.categorical_cols = ['brand', 'fuel_type', 'transmission', 'owner_type']
        self.target_col = 'price'
        self.irrelevant_cols = ['car_id', 'model']
        
        self.feature_columns = []
        self.imputation_values = {}
        self.outlier_bounds = {}
        self.metrics_summary = {}
        self.feature_importance = {}
        self.correlations = {}
        self.cleaning_log = {}

    # ==========================================
    # 1. DATA PREPROCESSING (DATA MINING TASKS)
    # ==========================================
    def preprocess_raw_data(self, df_raw):
        """
        Executes end-to-end Data Mining Preprocessing on raw dataset:
        - Logs initial state
        - Removes duplicates
        - Drops irrelevant attributes
        - Imputes missing values
        - Detects and treats outliers via IQR
        """
        df = df_raw.copy()
        raw_rows, raw_cols = df.shape
        
        # 1. Track missing values before imputation
        null_counts_before = df.isnull().sum().to_dict()
        total_nulls_before = int(df.isnull().sum().sum())
        
        # 2. Remove Duplicate Records
        duplicates_count = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)
        
        # 3. Drop irrelevant fields
        dropped_cols = [c for c in self.irrelevant_cols if c in df.columns]
        df = df.drop(columns=dropped_cols)
        
        # 4. Handle Missing Values (Imputation)
        # Numerical: Median (robust to skewness)
        # Categorical: Mode (most frequent category)
        imputation_log = {}
        for col in self.numerical_cols:
            if col in df.columns:
                median_val = float(df[col].median())
                self.imputation_values[col] = median_val
                null_cnt = int(df[col].isnull().sum())
                if null_cnt > 0:
                    df[col] = df[col].fillna(median_val)
                    imputation_log[col] = {'method': 'Median', 'imputed_value': round(median_val, 2), 'count': null_cnt}
                    
        for col in self.categorical_cols:
            if col in df.columns:
                mode_series = df[col].mode()
                mode_val = str(mode_series.iloc[0]) if not mode_series.empty else 'Unknown'
                self.imputation_values[col] = mode_val
                null_cnt = int(df[col].isnull().sum())
                if null_cnt > 0:
                    df[col] = df[col].fillna(mode_val)
                    imputation_log[col] = {'method': 'Mode', 'imputed_value': mode_val, 'count': null_cnt}
        
        if self.target_col in df.columns and df[self.target_col].isnull().sum() > 0:
            df = df.dropna(subset=[self.target_col]).reset_index(drop=True)

        null_counts_after = df.isnull().sum().to_dict()
        
        # 5. Outlier Detection & Treatment via IQR
        # Detect outliers using both IQR and Z-Score, treat using IQR capping (Winsorization)
        outlier_log = {}
        for col in ['mileage', 'horsepower', 'price']:
            if col in df.columns:
                # IQR calculation
                q1 = float(df[col].quantile(0.25))
                q3 = float(df[col].quantile(0.75))
                iqr = q3 - q1
                lower_bound = max(0.0, q1 - 1.5 * iqr)
                upper_bound = q3 + 1.5 * iqr
                
                # Outlier counts before
                iqr_outliers_count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
                
                # Z-score outlier count (|z| > 3)
                z_scores = np.abs(stats.zscore(df[col]))
                z_outliers_count = int((z_scores > 3).sum())
                
                self.outlier_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
                
                # Capping / Winsorization treatment
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                outlier_log[col] = {
                    'Q1': round(q1, 2),
                    'Q3': round(q3, 2),
                    'IQR': round(iqr, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'iqr_outliers_treated': iqr_outliers_count,
                    'z_score_outliers_count': z_outliers_count
                }
        
        self.cleaning_log = {
            'raw_shape': (raw_rows, raw_cols),
            'cleaned_shape': df.shape,
            'duplicates_removed': duplicates_count,
            'dropped_irrelevant_columns': dropped_cols,
            'total_nulls_before': total_nulls_before,
            'total_nulls_after': int(df.isnull().sum().sum()),
            'null_counts_before': null_counts_before,
            'null_counts_after': null_counts_after,
            'imputations': imputation_log,
            'outliers': outlier_log
        }
        
        return df

    # ==========================================
    # 2. FEATURE ENGINEERING & TRANSFORMATION
    # ==========================================
    def transform_features(self, df, is_training=True):
        """
        One-Hot Encodings for Categorical attributes,
        StandardScaler normalization for Numerical features.
        """
        # Separate X and y
        y = None
        if self.target_col in df.columns:
            y = df[self.target_col].values
            X_df = df.drop(columns=[self.target_col])
        else:
            X_df = df.copy()

        # One-Hot Encoding for categorical features
        cat_df = pd.get_dummies(X_df[self.categorical_cols], drop_first=True)
        
        # Numerical features
        num_df = X_df[self.numerical_cols].copy()
        
        if is_training:
            self.scaler = StandardScaler()
            scaled_num = self.scaler.fit_transform(num_df)
            scaled_num_df = pd.DataFrame(scaled_num, columns=self.numerical_cols, index=num_df.index)
            
            # Combine
            X_transformed = pd.concat([scaled_num_df, cat_df], axis=1)
            self.feature_columns = list(X_transformed.columns)
            return X_transformed, y
        else:
            scaled_num = self.scaler.transform(num_df)
            scaled_num_df = pd.DataFrame(scaled_num, columns=self.numerical_cols, index=num_df.index)
            X_combined = pd.concat([scaled_num_df, cat_df], axis=1)
            
            # Reindex to match trained feature columns precisely
            X_transformed = X_combined.reindex(columns=self.feature_columns, fill_value=0)
            return X_transformed, y

    # ==========================================
    # 3. FEATURE SELECTION & CORRELATION
    # ==========================================
    def analyze_correlations(self, df):
        """Computes Pearson correlations between numerical variables and target."""
        num_cols = [c for c in self.numerical_cols if c in df.columns]
        if self.target_col in df.columns:
            num_cols.append(self.target_col)
            
        corr_matrix = df[num_cols].corr().round(3)
        self.correlations = corr_matrix.to_dict()
        
        # Ranking against price
        if self.target_col in corr_matrix.columns:
            target_corr = corr_matrix[self.target_col].drop(self.target_col).sort_values(ascending=False).to_dict()
        else:
            target_corr = {}
            
        return corr_matrix, target_corr

    # ==========================================
    # 4. MODEL TRAINING & EVALUATION
    # ==========================================
    def train_models(self, X, y):
        """
        Trains and compares 5 ML algorithms:
        - Linear Regression
        - Ridge Regression
        - Decision Tree Regressor
        - Random Forest Regressor
        - Gradient Boosting Regressor
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Decision Tree Regressor': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1),
            'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)
        }
        
        results = []
        best_r2 = -float('inf')
        best_model_obj = None
        best_name = None
        
        n_test = len(y_test)
        p = X.shape[1]
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = float(r2_score(y_test, y_pred))
            adj_r2 = float(1 - (1 - r2) * (n_test - 1) / (n_test - p - 1)) if (n_test - p - 1) > 0 else r2
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100)
            
            results.append({
                'model_name': name,
                'r2_score': round(r2, 4),
                'adj_r2_score': round(adj_r2, 4),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'mape': round(mape, 2)
            })
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_obj = model
                best_name = name
                
        self.metrics_summary = results
        self.best_model_name = best_name
        self.best_model = best_model_obj
        
        # Calculate Feature Importances from Best Tree Model
        rf_model = models['Random Forest Regressor']
        importances = rf_model.feature_importances_
        feature_importance_dict = dict(zip(X.columns, [round(float(v), 4) for v in importances]))
        self.feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        # Save artifacts
        self.save_pipeline()
        
        return results, best_name

    # ==========================================
    # 5. EXPLORATORY DATA ANALYSIS (EDA PLOTS)
    # ==========================================
    def generate_eda_charts(self, df_cleaned):
        """Generates rich, interactive Plotly visualizations for the dashboard."""
        charts = {}
        
        # 1. Price Distribution Chart
        fig_price = px.histogram(
            df_cleaned, 
            x='price', 
            nbins=35, 
            marginal='box',
            title='Used Car Price Distribution (USD)',
            color_discrete_sequence=['#2563eb'],
            labels={'price': 'Car Price ($)'}
        )
        fig_price.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=50, b=20), height=380)
        charts['price_distribution'] = fig_price.to_html(full_html=False, include_plotlyjs=False)

        # 2. Mileage vs Price Scatter with Trendline
        fig_scatter = px.scatter(
            df_cleaned, 
            x='mileage', 
            y='price', 
            color='fuel_type', 
            trendline='ols',
            hover_data=['year', 'brand'],
            title='Mileage vs Price Dynamics by Fuel Type',
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={'mileage': 'Mileage (km)', 'price': 'Price ($)'}
        )
        fig_scatter.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=50, b=20), height=380)
        charts['mileage_vs_price'] = fig_scatter.to_html(full_html=False, include_plotlyjs=False)

        # 3. Average Price by Brand (Bar Chart)
        brand_summary = df_cleaned.groupby('brand')['price'].mean().reset_index().sort_values('price', ascending=False)
        fig_brand = px.bar(
            brand_summary, 
            x='brand', 
            y='price', 
            title='Market Valuation: Average Price by Brand',
            color='price',
            color_continuous_scale='Viridis',
            labels={'brand': 'Automobile Brand', 'price': 'Average Price ($)'}
        )
        fig_brand.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=50, b=20), height=380)
        charts['brand_price_bar'] = fig_brand.to_html(full_html=False, include_plotlyjs=False)

        # 4. Price by Transmission & Fuel Type (Box Plots)
        fig_box = px.box(
            df_cleaned, 
            x='fuel_type', 
            y='price', 
            color='transmission', 
            title='Price Distribution by Fuel Type & Transmission',
            color_discrete_sequence=['#3b82f6', '#ec4899'],
            labels={'fuel_type': 'Fuel Type', 'price': 'Price ($)'}
        )
        fig_box.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=50, b=20), height=380)
        charts['fuel_transmission_box'] = fig_box.to_html(full_html=False, include_plotlyjs=False)

        # 5. Correlation Heatmap
        corr_matrix, _ = self.analyze_correlations(df_cleaned)
        fig_heat = px.imshow(
            corr_matrix, 
            text_auto=True, 
            aspect='auto',
            color_continuous_scale='RdBu_r', 
            title='Pearson Correlation Heatmap (Numerical Attributes)',
            zmin=-1, zmax=1
        )
        fig_heat.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=50, b=20), height=380)
        charts['correlation_heatmap'] = fig_heat.to_html(full_html=False, include_plotlyjs=False)

        # 6. Feature Importance Bar Chart
        if self.feature_importance:
            top_feats = list(self.feature_importance.items())[:10]
            top_df = pd.DataFrame(top_feats, columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
            fig_imp = px.bar(
                top_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title='Top 10 Feature Importances (Random Forest)',
                color='Importance',
                color_continuous_scale='Plasma'
            )
            fig_imp.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=50, b=20), height=380)
            charts['feature_importance'] = fig_imp.to_html(full_html=False, include_plotlyjs=False)

        return charts

    # ==========================================
    # 6. SERIALIZATION & DESERIALIZATION
    # ==========================================
    def save_pipeline(self):
        """Saves models, scalers, and configuration metadata."""
        joblib.dump(self.best_model, os.path.join(self.models_dir, 'best_model.joblib'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.joblib'))
        
        meta = {
            'best_model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'imputation_values': self.imputation_values,
            'outlier_bounds': self.outlier_bounds,
            'metrics_summary': self.metrics_summary,
            'feature_importance': self.feature_importance,
            'cleaning_log': self.cleaning_log
        }
        with open(os.path.join(self.models_dir, 'pipeline_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def load_pipeline(self):
        """Loads trained artifacts from disk if available."""
        model_path = os.path.join(self.models_dir, 'best_model.joblib')
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        meta_path = os.path.join(self.models_dir, 'pipeline_meta.json')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path):
            self.best_model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.best_model_name = meta.get('best_model_name', 'Random Forest Regressor')
            self.feature_columns = meta.get('feature_columns', [])
            self.numerical_cols = meta.get('numerical_cols', self.numerical_cols)
            self.categorical_cols = meta.get('categorical_cols', self.categorical_cols)
            self.imputation_values = meta.get('imputation_values', {})
            self.outlier_bounds = meta.get('outlier_bounds', {})
            self.metrics_summary = meta.get('metrics_summary', [])
            self.feature_importance = meta.get('feature_importance', {})
            self.cleaning_log = meta.get('cleaning_log', {})
            return True
        return False

    # ==========================================
    # 7. REAL-TIME SINGLE SAMPLE INFERENCE
    # ==========================================
    def predict_car_price(self, brand, year, mileage, engine_size, horsepower, fuel_efficiency, seats, fuel_type, transmission, owner_type):
        """
        Takes human inputs from web form and returns predicted price + valuation range.
        """
        if self.best_model is None or self.scaler is None:
            loaded = self.load_pipeline()
            if not loaded:
                raise ValueError("Model pipeline is not yet trained. Run training first.")
                
        # Handle single sample DataFrame
        input_data = pd.DataFrame([{
            'brand': str(brand),
            'year': int(year),
            'mileage': float(mileage),
            'engine_size': float(engine_size),
            'horsepower': float(horsepower),
            'fuel_efficiency': float(fuel_efficiency),
            'seats': int(seats),
            'fuel_type': str(fuel_type),
            'transmission': str(transmission),
            'owner_type': str(owner_type)
        }])
        
        # Transform using training pipeline definition
        X_infer, _ = self.transform_features(input_data, is_training=False)
        
        # Predict
        predicted_price = float(self.best_model.predict(X_infer)[0])
        predicted_price = max(1500.0, round(predicted_price, 2))
        
        # Confidence interval approximation (~8% band based on test MAPE)
        low_estimate = round(predicted_price * 0.92, 2)
        high_estimate = round(predicted_price * 1.08, 2)
        
        return {
            'predicted_price': predicted_price,
            'low_estimate': low_estimate,
            'high_estimate': high_estimate,
            'model_used': self.best_model_name,
            'features_evaluated': input_data.to_dict(orient='records')[0]
        }


# ==========================================
# STANDALONE EXECUTION & HELPER FUNCTIONS
# ==========================================
def run_full_pipeline(dataset_path=DEFAULT_DATASET):
    """Executes the full Part 1 pipeline on given dataset path."""
    print("=" * 60)
    print("STARTING DATA MINING & ML PIPELINE (AUTOMOBILE ANALYTICS)")
    print("=" * 60)
    
    if dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
        df_raw = pd.read_excel(dataset_path)
    else:
        df_raw = pd.read_csv(dataset_path)
        
    pipeline = CarPricePipeline()
    
    print(f"\n[1/6] Loading raw dataset from: {dataset_path}")
    print(f"      Initial Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    
    print("\n[2/6] Data Preprocessing:")
    df_cleaned = pipeline.preprocess_raw_data(df_raw)
    log = pipeline.cleaning_log
    print(f"      - Duplicates Removed: {log['duplicates_removed']}")
    print(f"      - Irrelevant Columns Dropped: {log['dropped_irrelevant_columns']}")
    print(f"      - Missing Values Before: {log['total_nulls_before']} -> After: {log['total_nulls_after']}")
    print(f"      - Imputations performed: {len(log['imputations'])} columns")
    print(f"      - Outliers detected & capped via IQR: {list(log['outliers'].keys())}")
    print(f"      - Cleaned Shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
    
    print("\n[3/6] Feature Transformation (StandardScaler & One-Hot Encoding):")
    X, y = pipeline.transform_features(df_cleaned, is_training=True)
    print(f"      - Feature Matrix Shape: {X.shape}")
    print(f"      - Target Vector Length: {len(y)}")
    
    print("\n[4/6] Correlation Analysis & Feature Selection:")
    _, target_corrs = pipeline.analyze_correlations(df_cleaned)
    for feat, corr in list(target_corrs.items())[:5]:
        print(f"      - {feat:15s}: Pearson r = {corr:+.3f}")
        
    print("\n[5/6] Model Training & Performance Comparison (5 Algorithms):")
    results, best_model_name = pipeline.train_models(X, y)
    print(f"      {'Model':<30} | {'R²':<7} | {'Adj R²':<7} | {'RMSE':<9} | {'MAE':<8} | {'MAPE':<6}")
    print("      " + "-" * 75)
    for r in results:
        print(f"      {r['model_name']:<30} | {r['r2_score']:<7.4f} | {r['adj_r2_score']:<7.4f} | {r['rmse']:<9.2f} | {r['mae']:<8.2f} | {r['mape']:<6.2f}%")
    print(f"\n      >>> Selected Best Model: {best_model_name} <<<")
    
    print("\n[6/6] Generating EDA Visualizations & Serializing Models...")
    charts = pipeline.generate_eda_charts(df_cleaned)
    print(f"      Generated {len(charts)} interactive Plotly visualizations.")
    print(f"      Saved model and artifacts to {pipeline.models_dir}")
    
    print("\nPipeline execution completed successfully!")
    print("=" * 60)
    return pipeline, df_cleaned, charts


if __name__ == '__main__':
    run_full_pipeline()
