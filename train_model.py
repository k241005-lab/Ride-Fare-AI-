import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def build_dataset():
    np.random.seed(42)
    n_samples = 10000
    df = pd.DataFrame({
        'pickup':     np.random.choice(['downtown', 'airport', 'suburb', 'midtown', 'uptown'], n_samples),
        'dropoff':    np.random.choice(['airport', 'downtown', 'suburb', 'midtown', 'uptown'], n_samples),
        'distance':   np.random.uniform(1, 80, n_samples),
        'passengers': np.random.randint(1, 7, n_samples),
        'timeofday':  np.random.choice(['morning', 'midday', 'evening', 'night', 'latenight'], n_samples),
        'traffic':    np.random.choice(['low', 'medium', 'high', 'jam'], n_samples),
        'ridetype':   np.random.choice(['standard', 'xl', 'premium', 'economy'], n_samples)
    })
    base_fare = 200
    df['fare_amount'] = base_fare + (df['distance'] * 80)
    df.loc[df['traffic'] == 'jam',       'fare_amount'] *= 1.8
    df.loc[df['ridetype'] == 'premium',  'fare_amount'] *= 2.0
    df['fare_amount'] += np.random.normal(0, 150, n_samples)
    df['fare_amount'] = df['fare_amount'].clip(lower=200)
    return df

def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(),                     ['distance', 'passengers']),
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         ['pickup', 'dropoff', 'timeofday', 'traffic', 'ridetype'])
    ])

def train_and_save_all():
    print("=" * 55)
    print("  RideFare AI - Training All 3 Models")
    print("=" * 55)

    print("\n[1/5] Building dataset...")
    df = build_dataset()
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"      Train: {len(X_train):,} samples  |  Test: {len(X_test):,} samples")

    models = {
        'random_forest':      RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
        'linear_regression':  LinearRegression(),
    }

    results = {}

    for i, (key, regressor) in enumerate(models.items(), start=2):
        name = key.replace('_', ' ').title()
        print(f"\n[{i}/5] Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('regressor',    regressor)
        ])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        print(f"      MAE  : PKR {mae:.0f}")
        print(f"      RMSE : PKR {rmse:.0f}")
        print(f"      R2   : {r2:.4f}")

        filename = f'{key}_model.pkl'
        joblib.dump(pipeline, filename)
        print(f"      Saved: {filename}")

        results[key] = {'mae': round(mae), 'rmse': round(rmse), 'r2': round(r2, 4)}

    # Save the best model as the default
    print("\n[5/5] Saving best model (Random Forest) as default taxi_fare_model.pkl ...")
    best = joblib.load('random_forest_model.pkl')
    joblib.dump(best, 'taxi_fare_model.pkl')

    print("\n" + "=" * 55)
    print("  All models saved successfully!")
    print("  Files created:")
    print("    - random_forest_model.pkl")
    print("    - gradient_boosting_model.pkl")
    print("    - linear_regression_model.pkl")
    print("    - taxi_fare_model.pkl (default)")
    print("=" * 55)

    print("\n  Model Comparison Summary:")
    print(f"  {'Model':<25} {'MAE (PKR)':>10} {'RMSE (PKR)':>12} {'R2':>8}")
    print("  " + "-" * 57)
    
    labels = {
        'random_forest':     'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
        'linear_regression': 'Linear Regression',
    }
    
    for key, r in results.items():
        print(f"  {labels[key]:<25} {r['mae']:>10,} {r['rmse']:>12,} {r['r2']:>8.4f}")
    print()

if __name__ == "__main__":
    train_and_save_all()