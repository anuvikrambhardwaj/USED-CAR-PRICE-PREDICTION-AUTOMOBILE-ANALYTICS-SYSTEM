"""
Script to generate a rich, realistic Used Car Price dataset (1,250 records)
with both numerical and categorical attributes, containing realistic missing values,
duplicates, and outliers for Data Mining & Machine Learning demonstrations.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_samples = 1250

brands_data = {
    'Toyota': {'models': ['Corolla', 'Camry', 'RAV4', 'Yaris', 'Fortuner'], 'base_price': 18000, 'hp_range': (100, 220), 'engine_range': (1.4, 2.5)},
    'Honda': {'models': ['Civic', 'Accord', 'CR-V', 'City', 'Fit'], 'base_price': 17500, 'hp_range': (95, 200), 'engine_range': (1.2, 2.4)},
    'Ford': {'models': ['Focus', 'Mustang', 'Explorer', 'Fiesta', 'F-150'], 'base_price': 20000, 'hp_range': (120, 380), 'engine_range': (1.5, 4.0)},
    'BMW': {'models': ['3 Series', '5 Series', 'X3', 'X5', 'M3'], 'base_price': 34000, 'hp_range': (180, 420), 'engine_range': (2.0, 3.5)},
    'Mercedes-Benz': {'models': ['C-Class', 'E-Class', 'GLC', 'GLE', 'A-Class'], 'base_price': 36000, 'hp_range': (170, 430), 'engine_range': (2.0, 3.8)},
    'Audi': {'models': ['A3', 'A4', 'A6', 'Q3', 'Q5'], 'base_price': 33000, 'hp_range': (150, 360), 'engine_range': (1.8, 3.0)},
    'Hyundai': {'models': ['Elantra', 'Sonata', 'Tucson', 'Santa Fe', 'i20'], 'base_price': 15000, 'hp_range': (85, 190), 'engine_range': (1.2, 2.2)},
    'Nissan': {'models': ['Altima', 'Sentra', 'Rogue', 'Qashqai', '370Z'], 'base_price': 16000, 'hp_range': (90, 280), 'engine_range': (1.3, 3.2)},
    'Chevrolet': {'models': ['Cruze', 'Malibu', 'Camaro', 'Equinox', 'Tahoe'], 'base_price': 19000, 'hp_range': (110, 350), 'engine_range': (1.4, 4.0)},
    'Volkswagen': {'models': ['Golf', 'Passat', 'Tiguan', 'Polo', 'Jetta'], 'base_price': 17000, 'hp_range': (90, 220), 'engine_range': (1.2, 2.5)},
}

brands_list = list(brands_data.keys())
brand_weights = [0.18, 0.16, 0.12, 0.08, 0.07, 0.07, 0.14, 0.08, 0.05, 0.05]

selected_brands = np.random.choice(brands_list, size=n_samples, p=brand_weights)

rows = []
for i, brand in enumerate(selected_brands):
    info = brands_data[brand]
    model = np.random.choice(info['models'])
    year = int(np.random.choice(range(2010, 2025), p=[0.04, 0.04, 0.05, 0.06, 0.07, 0.07, 0.08, 0.09, 0.10, 0.10, 0.10, 0.08, 0.06, 0.04, 0.02]))
    
    # Age effect
    age = 2025 - year
    mileage = int(max(4000, np.random.normal(age * 12500, 18000)))
    
    fuel_type = np.random.choice(['Petrol', 'Diesel', 'Hybrid', 'Electric'], p=[0.58, 0.24, 0.14, 0.04])
    transmission = np.random.choice(['Automatic', 'Manual'], p=[0.68, 0.32])
    owner_type = np.random.choice(['First', 'Second', 'Third'], p=[0.62, 0.28, 0.10])
    
    engine_size = round(np.random.uniform(info['engine_range'][0], info['engine_range'][1]), 1)
    if fuel_type == 'Electric':
        engine_size = 0.0
        
    horsepower = int(np.random.uniform(info['hp_range'][0], info['hp_range'][1]))
    fuel_efficiency = round(max(9.0, np.random.normal(26.0 - (engine_size * 2.8) + (2.0 if fuel_type == 'Diesel' else 0) + (5.0 if fuel_type == 'Hybrid' else 0), 3.2)), 1)
    seats = int(np.random.choice([4, 5, 7], p=[0.08, 0.82, 0.10]))
    
    # Price estimation formula with realistic market dynamics + random noise
    base = info['base_price']
    depreciation = (1 - 0.065) ** age
    mileage_factor = max(0.45, 1.0 - (mileage / 280000))
    hp_bonus = (horsepower - 100) * 85
    trans_bonus = 1500 if transmission == 'Automatic' else 0
    owner_discount = 0.88 if owner_type == 'Second' else (0.75 if owner_type == 'Third' else 1.0)
    fuel_adj = 1.12 if fuel_type == 'Hybrid' else (1.20 if fuel_type == 'Electric' else (1.05 if fuel_type == 'Diesel' else 1.0))
    
    price = (base * depreciation * mileage_factor + hp_bonus + trans_bonus) * owner_discount * fuel_adj
    price += np.random.normal(0, 1400)
    price = max(2800, round(price, -2))
    
    rows.append({
        'car_id': f'CAR_{1000 + i}',
        'brand': brand,
        'model': model,
        'year': year,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'owner_type': owner_type,
        'mileage': mileage,
        'engine_size': engine_size,
        'horsepower': horsepower,
        'fuel_efficiency': fuel_efficiency,
        'seats': seats,
        'price': price
    })

df = pd.DataFrame(rows)

# Inject controlled Data Mining artifacts:
# 1. Duplicate records (~15 duplicates)
duplicates = df.sample(n=15, random_state=123)
df = pd.concat([df, duplicates], ignore_index=True)

# 2. Missing values (~3% in numerical and categorical fields)
nan_idx_mileage = np.random.choice(df.index, size=35, replace=False)
df.loc[nan_idx_mileage, 'mileage'] = np.nan

nan_idx_engine = np.random.choice(df.index, size=28, replace=False)
df.loc[nan_idx_engine, 'engine_size'] = np.nan

nan_idx_fuel = np.random.choice(df.index, size=22, replace=False)
df.loc[nan_idx_fuel, 'fuel_type'] = np.nan

nan_idx_hp = np.random.choice(df.index, size=25, replace=False)
df.loc[nan_idx_hp, 'horsepower'] = np.nan

nan_idx_seats = np.random.choice(df.index, size=15, replace=False)
df.loc[nan_idx_seats, 'seats'] = np.nan

# 3. Controlled Outliers (extreme mileage or price anomalies for IQR demonstration)
outlier_idx_mileage = np.random.choice(df.index, size=8, replace=False)
df.loc[outlier_idx_mileage, 'mileage'] = df.loc[outlier_idx_mileage, 'mileage'] * 3.5 + 150000

outlier_idx_price = np.random.choice(df.index, size=7, replace=False)
df.loc[outlier_idx_price, 'price'] = df.loc[outlier_idx_price, 'price'] * 3.2 + 25000

os.makedirs('data', exist_ok=True)
csv_path = os.path.join('data', 'used_cars_dataset.csv')
df.to_csv(csv_path, index=False)
print(f"Dataset successfully created at: {csv_path}")
print(f"Total shape: {df.shape}")
print(f"Duplicates count: 15")
print(f"Missing values count:\n{df.isnull().sum()}")
