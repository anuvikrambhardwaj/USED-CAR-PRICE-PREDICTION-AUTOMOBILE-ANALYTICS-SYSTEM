"""
Views for Used Car Price Prediction & Automobile Analytics System
"""

import os
import json
import pandas as pd
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib import messages

from .ml_pipeline import CarPricePipeline, run_full_pipeline, DEFAULT_DATASET, MODELS_DIR

# Initialize global pipeline instance
pipeline_instance = CarPricePipeline(models_dir=MODELS_DIR)
# Attempt to load trained pipeline on startup
if not pipeline_instance.load_pipeline():
    if os.path.exists(DEFAULT_DATASET):
        pipeline_instance, _, _ = run_full_pipeline(DEFAULT_DATASET)


def get_current_dataset_path(request):
    """Retrieves path to active dataset (user-uploaded or default Kaggle sample)."""
    dataset_path = request.session.get('active_dataset_path')
    if dataset_path and os.path.exists(dataset_path):
        return dataset_path
    return DEFAULT_DATASET


def home_view(request):
    """1. File Upload Interface & Welcome Landing Page."""
    current_path = get_current_dataset_path(request)
    is_custom = (current_path != DEFAULT_DATASET)
    dataset_name = os.path.basename(current_path)
    
    meta_path = os.path.join(MODELS_DIR, 'pipeline_meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
    context = {
        'dataset_name': dataset_name,
        'is_custom': is_custom,
        'meta': meta,
        'best_model_name': meta.get('best_model_name', 'Ridge / Random Forest Regressor'),
    }
    return render(request, 'analytics/index.html', context)


def upload_dataset_view(request):
    """Processes uploaded dataset (.csv or .xlsx) or resets to default."""
    global pipeline_instance
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'load_default':
            request.session['active_dataset_path'] = DEFAULT_DATASET
            pipeline_instance.load_pipeline()
            messages.success(request, "Default Kaggle Used Cars Dataset loaded successfully (1,265 records).")
            return redirect('data_summary')
            
        uploaded_file = request.FILES.get('dataset_file')
        if not uploaded_file:
            messages.error(request, "Please select a valid CSV or Excel file to upload.")
            return redirect('home')
            
        file_name = uploaded_file.name
        if not (file_name.endswith('.csv') or file_name.endswith('.xlsx') or file_name.endswith('.xls')):
            messages.error(request, "Unsupported file format. Please upload a .csv or .xlsx dataset.")
            return redirect('home')
            
        # Save file to media directory
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, file_name)
        
        with open(save_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
                
        # Validate dataset structure
        try:
            if file_name.endswith('.csv'):
                df_test = pd.read_csv(save_path)
            else:
                df_test = pd.read_excel(save_path)
                
            required_cols = ['year', 'mileage', 'price', 'brand', 'fuel_type', 'transmission']
            missing_req = [c for c in required_cols if c not in df_test.columns]
            if missing_req:
                messages.warning(request, f"Uploaded dataset is missing expected columns: {', '.join(missing_req)}. Falling back to default dataset.")
                return redirect('home')
                
            # Run pipeline on uploaded dataset
            pipeline_instance, _, _ = run_full_pipeline(save_path)
            request.session['active_dataset_path'] = save_path
            messages.success(request, f"Dataset '{file_name}' uploaded, preprocessed, and ML models trained successfully!")
            return redirect('data_summary')
            
        except Exception as e:
            messages.error(request, f"Error processing uploaded dataset: {str(e)}")
            return redirect('home')
            
    return redirect('home')


def data_summary_view(request):
    """2. Data Summary Preview: Shape, Top 10 Preview, Null Imputations, Outlier Treatment, Stats."""
    current_path = get_current_dataset_path(request)
    
    # Read raw data
    if current_path.endswith('.xlsx') or current_path.endswith('.xls'):
        df_raw = pd.read_excel(current_path)
    else:
        df_raw = pd.read_csv(current_path)
        
    raw_shape = df_raw.shape
    raw_null_counts = df_raw.isnull().sum().to_dict()
    total_raw_nulls = int(df_raw.isnull().sum().sum())
    
    # Preprocess via pipeline
    df_cleaned = pipeline_instance.preprocess_raw_data(df_raw)
    cleaning_log = pipeline_instance.cleaning_log
    
    # Preview tables
    preview_df = df_cleaned.head(10).round(2)
    preview_rows = preview_df.to_dict(orient='records')
    preview_columns = preview_df.columns.tolist()
    
    # Describe statistics
    stats_df = df_cleaned.describe().round(2).reset_index()
    stats_rows = stats_df.to_dict(orient='records')
    stats_columns = stats_df.columns.tolist()
    
    # Data types table
    dtypes_list = [{'column': col, 'dtype': str(dtype), 'non_null': int(df_cleaned[col].count()), 'null_count': int(df_cleaned[col].isnull().sum())} for col, dtype in df_cleaned.dtypes.items()]
    
    context = {
        'dataset_name': os.path.basename(current_path),
        'raw_shape': raw_shape,
        'cleaned_shape': df_cleaned.shape,
        'duplicates_removed': cleaning_log.get('duplicates_removed', 0),
        'dropped_cols': cleaning_log.get('dropped_irrelevant_columns', []),
        'total_raw_nulls': total_raw_nulls,
        'imputations': cleaning_log.get('imputations', {}),
        'outliers': cleaning_log.get('outliers', {}),
        'preview_columns': preview_columns,
        'preview_rows': preview_rows,
        'stats_columns': stats_columns,
        'stats_rows': stats_rows,
        'dtypes_list': dtypes_list,
    }
    return render(request, 'analytics/data_summary.html', context)


def visual_dashboard_view(request):
    """3. Visual Analytics Dashboard: Dynamic Interactive Plotly Visualizations."""
    current_path = get_current_dataset_path(request)
    
    if current_path.endswith('.xlsx') or current_path.endswith('.xls'):
        df = pd.read_excel(current_path)
    else:
        df = pd.read_csv(current_path)
        
    df_cleaned = pipeline_instance.preprocess_raw_data(df)
    charts = pipeline_instance.generate_eda_charts(df_cleaned)
    
    # Additional high-level KPI cards
    kpis = {
        'total_records': len(df_cleaned),
        'avg_price': round(float(df_cleaned['price'].mean()), 2),
        'median_price': round(float(df_cleaned['price'].median()), 2),
        'avg_mileage': round(float(df_cleaned['mileage'].mean()), 0),
        'top_brand': df_cleaned['brand'].value_counts().index[0],
        'most_common_fuel': df_cleaned['fuel_type'].value_counts().index[0],
    }
    
    context = {
        'charts': charts,
        'kpis': kpis,
        'dataset_name': os.path.basename(current_path)
    }
    return render(request, 'analytics/dashboard.html', context)


def predict_view(request):
    """4. Prediction / Result Section: Model Performance Cards + Real-time Form."""
    meta_path = os.path.join(MODELS_DIR, 'pipeline_meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
    metrics_summary = meta.get('metrics_summary', pipeline_instance.metrics_summary)
    best_model_name = meta.get('best_model_name', pipeline_instance.best_model_name)
    feature_importance = meta.get('feature_importance', {})
    
    prediction_result = None
    form_inputs = {
        'brand': 'Toyota',
        'year': 2021,
        'mileage': 35000,
        'engine_size': 2.0,
        'horsepower': 160,
        'fuel_efficiency': 18.0,
        'seats': 5,
        'fuel_type': 'Petrol',
        'transmission': 'Automatic',
        'owner_type': 'First'
    }
    
    if request.method == 'POST':
        try:
            form_inputs = {
                'brand': request.POST.get('brand', 'Toyota'),
                'year': int(request.POST.get('year', 2021)),
                'mileage': float(request.POST.get('mileage', 35000)),
                'engine_size': float(request.POST.get('engine_size', 2.0)),
                'horsepower': float(request.POST.get('horsepower', 160)),
                'fuel_efficiency': float(request.POST.get('fuel_efficiency', 18.0)),
                'seats': int(request.POST.get('seats', 5)),
                'fuel_type': request.POST.get('fuel_type', 'Petrol'),
                'transmission': request.POST.get('transmission', 'Automatic'),
                'owner_type': request.POST.get('owner_type', 'First')
            }
            
            prediction_result = pipeline_instance.predict_car_price(
                brand=form_inputs['brand'],
                year=form_inputs['year'],
                mileage=form_inputs['mileage'],
                engine_size=form_inputs['engine_size'],
                horsepower=form_inputs['horsepower'],
                fuel_efficiency=form_inputs['fuel_efficiency'],
                seats=form_inputs['seats'],
                fuel_type=form_inputs['fuel_type'],
                transmission=form_inputs['transmission'],
                owner_type=form_inputs['owner_type']
            )
        except Exception as e:
            messages.error(request, f"Error calculating prediction: {str(e)}")

    context = {
        'metrics_summary': metrics_summary,
        'best_model_name': best_model_name,
        'feature_importance': list(feature_importance.items())[:8],
        'prediction_result': prediction_result,
        'form_inputs': form_inputs,
    }
    return render(request, 'analytics/predict.html', context)


@csrf_exempt
def api_predict_view(request):
    """Async API endpoint for real-time interactive predictions without page reload."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            result = pipeline_instance.predict_car_price(
                brand=data.get('brand', 'Toyota'),
                year=int(data.get('year', 2021)),
                mileage=float(data.get('mileage', 35000)),
                engine_size=float(data.get('engine_size', 2.0)),
                horsepower=float(data.get('horsepower', 160)),
                fuel_efficiency=float(data.get('fuel_efficiency', 18.0)),
                seats=int(data.get('seats', 5)),
                fuel_type=data.get('fuel_type', 'Petrol'),
                transmission=data.get('transmission', 'Automatic'),
                owner_type=data.get('owner_type', 'First')
            )
            return JsonResponse({'status': 'success', 'data': result})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Only POST method allowed'}, status=405)


def lab_report_view(request):
    """Bonus Lab Project: Comprehensive Data Mining Lab Report and Architecture."""
    meta_path = os.path.join(MODELS_DIR, 'pipeline_meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
    context = {
        'meta': meta,
        'metrics_summary': meta.get('metrics_summary', []),
        'cleaning_log': meta.get('cleaning_log', {}),
        'best_model_name': meta.get('best_model_name', 'Ridge / Random Forest Regressor')
    }
    return render(request, 'analytics/lab_report.html', context)
