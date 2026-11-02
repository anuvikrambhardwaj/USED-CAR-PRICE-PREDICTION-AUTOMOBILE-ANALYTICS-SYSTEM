# Used Car Price Prediction & Automobile Analytics System
## Data Mining & Machine Learning Lab Project

An end-to-end Data Mining pipeline and interactive Django Web Application engineered for automated automobile valuation, exploratory data analytics, outlier handling, and multi-model machine learning comparison.

---

### Project Architecture & Features

#### Part 1: Data Mining & ML Pipeline (`analytics/ml_pipeline.py`)
1. **Dataset Selection**:
   - Bundled with a 1,265+ record Kaggle-style automobile dataset containing numerical attributes (`year`, `mileage`, `engine_size`, `horsepower`, `fuel_efficiency`, `seats`, `price`) and categorical attributes (`brand`, `fuel_type`, `transmission`, `owner_type`).
2. **Data Preprocessing Techniques**:
   - **Missing Value Identification & Imputation**: Median imputation for numerical fields (robust against distribution skewness); Mode imputation for categorical features.
   - **Outlier Detection & Treatment**: IQR (Interquartile Range) method ($[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$) with Winsorization (capping) plus Z-score statistics ($|z| > 3$).
   - **Data Cleaning**: Removal of duplicate rows and irrelevant identifier attributes (`car_id`, `model`).
   - **Data Transformation**: Feature normalization via `StandardScaler` ($z = \frac{x - \mu}{\sigma}$) and One-Hot Encoding (`get_dummies`) for nominal categorical features.
   - **Feature Selection**: Correlation matrix analysis with target (`price`) and Feature Importance ranking using Random Forest ensembles.
3. **Exploratory Data Analysis (EDA) & Visualization**:
   - Price distribution histogram & box plot
   - Mileage vs. Price scatter plot with OLS regression trendline
   - Brand valuation bar chart
   - Fuel Type & Transmission box plots (IQR spread)
   - Pearson Correlation Heatmap
   - Random Forest Feature Importance chart
4. **Machine Learning Model Training & Evaluation**:
   - Evaluates 5 regression models:
     1. Linear Regression (Baseline)
     2. Ridge Regression (L2 Regularized)
     3. Decision Tree Regressor
     4. Random Forest Regressor (Ensemble)
     5. Gradient Boosting Regressor
   - Metrics computed: $R^2$ Score, Adjusted $R^2$, RMSE, MAE, MAPE.
   - Serializes trained artifacts with `joblib`.

---

#### Part 2: Django Web Application (User Interface)
1. **File Upload Interface (`/`)**:
   - Drag-and-drop or file selector accepting `.csv` or `.xlsx` datasets.
   - One-click "Load Default Kaggle Dataset" button.
2. **Data Summary Preview (`/data-summary/`)**:
   - Shape overview before and after cleaning.
   - Missing values imputation table (method, imputed value, counts).
   - IQR outlier treatment breakdown (Q1, Q3, bounds, count of capped outliers).
   - Cleaned dataset top 10 preview table.
   - Descriptive statistics (`describe()` table with mean, std, quartiles).
3. **Visual Analytics Dashboard (`/dashboard/`)**:
   - 6 dynamic interactive Plotly charts with zoom, hover tooltips, and pan controls.
   - High-level KPI cards (Mean Price, Median Price, Average Mileage, Top Brand).
4. **Prediction & Result Section (`/predict/`)**:
   - Machine Learning models comparison scorecard.
   - Interactive valuation form (Brand, Year, Mileage, Engine Size, Horsepower, Fuel, Transmission, Owner Type).
   - Instant real-time AJAX prediction with confidence intervals ($\pm 8\%$).
5. **Academic Lab Report (`/lab-report/`)**:
   - Theoretical methodology, mathematical formulas, and Viva-Voce preparation guide.

---

### How to Run the Project

1. Open PowerShell or Terminal and navigate to the project directory:
   ```bash
   cd C:\Users\vikra\.gemini\antigravity\scratch\car_price_analytics
   ```

2. Activate or use Python with the installed dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Run the Data Mining Pipeline standalone from the terminal:
   ```bash
   python analytics/ml_pipeline.py
   ```

4. Start the Django Web Application:
   ```bash
   python manage.py runserver 127.0.0.1:8000
   ```

5. Open your web browser and navigate to:
   - **Upload & Home**: `http://127.0.0.1:8000/`
   - **Data Summary Preview**: `http://127.0.0.1:8000/data-summary/`
   - **Visual Analytics Dashboard**: `http://127.0.0.1:8000/dashboard/`
   - **Real-Time Price Prediction**: `http://127.0.0.1:8000/predict/`
   - **Lab Project Report**: `http://127.0.0.1:8000/lab-report/`
