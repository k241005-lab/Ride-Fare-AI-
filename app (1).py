from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')

# ── Load all three trained models ──────────────────────────────────────────
MODELS = {}
MODEL_META = {
    'random_forest': {
        'label':  'Random Forest',
        'mae':    129,
        'rmse':   161,
        'r2':     0.998,
        'color':  '#00d2ff',
        'confidence': 94,
    },
    'gradient_boosting': {
        'label':  'Gradient Boosting',
        'mae':    141,
        'rmse':   187,
        'r2':     0.998,
        'color':  '#7b2fff',
        'confidence': 92,
    },
    'linear_regression': {
        'label':  'Linear Regression',
        'mae':    957,
        'rmse':   1418,
        'r2':     0.874,
        'color':  '#ffb300',
        'confidence': 74,
    },
}

for key in MODEL_META:
    try:
        MODELS[key] = joblib.load(f'{key}_model.pkl')
        print(f"[OK] Loaded {MODEL_META[key]['label']}")
    except FileNotFoundError:
        print(f"[ERROR] {key}_model.pkl not found - run train_model.py first!")

# ── Helper ─────────────────────────────────────────────────────────────────
def build_input_df(data):
    return pd.DataFrame([{
        'pickup':     data['pickup'],
        'dropoff':    data['dropoff'],
        'distance':   float(data['distance']),
        'passengers': int(data['passengers']),
        'timeofday':  data['timeofday'],
        'traffic':    data['traffic'],
        'ridetype':   data['ridetype'],
    }])

def build_result(model_key, fare, distance):
    meta     = MODEL_META[model_key]
    duration = int(float(distance) * 2.5)
    per_km   = fare / float(distance)
    return {
        'model':      model_key,
        'label':      meta['label'],
        'color':      meta['color'],
        'fare':       round(fare),
        'duration':   f'~{duration} mins',
        'per_km':     f"PKR {round(per_km)}/km",
        'confidence': meta['confidence'],
        'mae':        meta['mae'],
        'rmse':       meta['rmse'],
        'r2':         meta['r2'],
    }

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('uber_trip_cost_predictor.html')

@app.route('/predict', methods=['POST'])
def predict_fare():
    """Single-model prediction. Pass 'model' key in JSON (default: random_forest)."""
    try:
        data       = request.json
        model_key  = data.get('model', 'random_forest')
        surge      = '1.5x' if data['traffic'] in ['high', 'jam'] else '1.0x'

        if model_key not in MODELS:
            return jsonify({'success': False, 'error': f'Model "{model_key}" not loaded.'})

        input_df = build_input_df(data)
        fare     = max(MODELS[model_key].predict(input_df)[0], 200)  # min fare PKR 200
        result   = build_result(model_key, fare, data['distance'])
        result['success'] = True
        result['surge']   = surge
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/compare', methods=['POST'])
def compare_models():
    """Returns predictions from all three models for side-by-side comparison."""
    try:
        data     = request.json
        surge    = '1.5x' if data['traffic'] in ['high', 'jam'] else '1.0x'
        input_df = build_input_df(data)

        results = []
        for key, model in MODELS.items():
            fare   = max(model.predict(input_df)[0], 200)  # min fare PKR 200
            res    = build_result(key, fare, data['distance'])
            res['surge'] = surge
            results.append(res)

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_best', methods=['POST'])
def predict_best():
    """Finds and returns the best model's prediction (lowest fare)."""
    try:
        data     = request.json
        surge    = '1.5x' if data['traffic'] in ['high', 'jam'] else '1.0x'
        input_df = build_input_df(data)

        # Find the model that predicts the lowest fare
        best_model_key = None
        min_fare = float('inf')

        for key in MODELS:
            fare = float(MODELS[key].predict(input_df)[0])
            if fare < min_fare:
                min_fare = fare
                best_model_key = key

        # If somehow no model key was found, default to random_forest
        if not best_model_key:
            best_model_key = 'random_forest'

        fare   = max(min_fare, 200)  # min fare PKR 200
        res    = build_result(best_model_key, fare, data['distance'])
        
        # Add required extra fields for the frontend
        res['best_model_name'] = res['label']
        res['ridetype'] = data['ridetype']
        res['passengers'] = data['passengers']
        res['success'] = True
        
        return jsonify(res)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/models', methods=['GET'])
def list_models():
    """Returns available model metadata for the frontend."""
    return jsonify({
        key: {'label': v['label'], 'mae': v['mae'], 'rmse': v['rmse'], 'r2': v['r2'], 'color': v['color']}
        for key, v in MODEL_META.items()
        if key in MODELS
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
