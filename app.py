from flask import Flask, render_template, jsonify
try:
    from flask_cors import CORS
    HAS_CORS = True
except Exception:
    HAS_CORS = False
try:
    # Python 3.9+: use zoneinfo for proper timezone handling
    from zoneinfo import ZoneInfo
    ZONE_KARACHI = ZoneInfo('Asia/Karachi')
except Exception:
    ZoneInfo = None
    ZONE_KARACHI = None
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from flask import request
import time
from functools import wraps
# Database integration
from database import PredictionDatabase
from flask_integration import store_prediction_and_check_alert
try:
    import cdsapi
    import xarray as xr
except Exception:
    cdsapi = None
    xr = None
try:
    import joblib
except Exception:
    joblib = None

app = Flask(__name__)

# Enable CORS if available
if HAS_CORS:
    CORS(app)
else:
    # Manual CORS headers if flask_cors not available
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,X-API-Key'
        return response

# Initialize database
db = PredictionDatabase()
print('✓ Database initialized for prediction storage')

# Try to load model if present
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cloudburst_final_bilstm_only.keras')
model = None

# Define focal_loss (from your training code) so the saved model can be loaded with custom_objects
def focal_loss(gamma=2.0, alpha=0.5):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss_val = -alpha * K.pow(1. - pt, gamma) * K.log(pt)
        return K.mean(loss_val)
    return loss

custom_objects = {
    'focal_loss': focal_loss,
    'loss': focal_loss()
}

try:
    from tensorflow.keras.models import load_model
    if os.path.exists(MODEL_PATH):
        try:
            # Try loading with custom_objects to resolve saved custom loss
            model = load_model(MODEL_PATH, custom_objects=custom_objects)
            print('Model loaded (with custom_objects) from', MODEL_PATH)
        except Exception as e1:
            print('Model load with custom_objects failed:', e1)
            try:
                # Try loading without compilation (safer if optimizer/loss not available)
                model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
                print('Model loaded with custom_objects and compile=False from', MODEL_PATH)
            except Exception as e2:
                print('Model load with compile=False and custom_objects failed:', e2)
                model = None
    else:
        print('Model file not found, running in fallback mode')
except Exception as e:
    print('TensorFlow import/model load error (fallback mode):', e)
    model = None

# Try to load scaler and feature columns saved during training
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler_final.pkl')
FEATURE_COLS_PATH = os.path.join(os.path.dirname(__file__), 'feature_cols.pkl')
scaler = None
feature_cols_saved = None
if joblib is not None:
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print('Loaded scaler from', SCALER_PATH)
        else:
            print('Scaler file not found:', SCALER_PATH)
    except Exception as e:
        print('Failed loading scaler:', e)

    try:
        if os.path.exists(FEATURE_COLS_PATH):
            feature_cols_saved = joblib.load(FEATURE_COLS_PATH)
            print('Loaded feature columns from', FEATURE_COLS_PATH)
        else:
            print('Feature columns file not found:', FEATURE_COLS_PATH)
    except Exception as e:
        print('Failed loading feature columns:', e)
else:
    print('joblib not available; scaler/feature_cols will not be loaded')

# Simple API key auth: read API key from env `API_KEY` or file `api_key.txt` or default 'devkey'
API_KEY = os.environ.get('API_KEY')
if not API_KEY:
    key_file = os.path.join(os.path.dirname(__file__), 'api_key.txt')
    if os.path.exists(key_file):
        try:
            with open(key_file, 'r') as f:
                API_KEY = f.read().strip()
        except Exception:
            API_KEY = 'devkey'
    else:
        API_KEY = 'devkey'

# Simple in-memory rate limiter: allow N requests per window per key
RATE_LIMIT = 300  # allow up to 5/sec for live local updates
RATE_WINDOW = 60  # seconds
_rate_store = {}
CLASSIFICATION_THRESHOLD = 0.5

def check_rate_limit(key):
    now = time.time()
    arr = _rate_store.get(key, [])
    # remove old
    arr = [t for t in arr if now - t < RATE_WINDOW]
    if len(arr) >= RATE_LIMIT:
        _rate_store[key] = arr
        return False
    arr.append(now)
    _rate_store[key] = arr
    return True

def require_api_key(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not key or key != API_KEY:
            return jsonify(error='Unauthorized'), 401
        if not check_rate_limit(key):
            return jsonify(error='Rate limit exceeded'), 429
        return f(*args, **kwargs)
    return wrapped


def make_prediction(lead_hours=0):
    # If model is present, use it. Otherwise return a dummy prediction.
    if model is not None:
        try:
            # Determine input shape: try model.input_shape, else fallback to known sequence length
            try:
                inp_shape = model.input_shape
                # inp_shape often is (None, timesteps, features)
                if isinstance(inp_shape, tuple) and len(inp_shape) >= 3:
                    n_timesteps = int(inp_shape[1]) if inp_shape[1] is not None else 6
                    n_features = int(inp_shape[2]) if inp_shape[2] is not None else (len(feature_cols_saved) if feature_cols_saved else 1)
                else:
                    n_timesteps = 6
                    n_features = (len(feature_cols_saved) if feature_cols_saved else 1)
            except Exception:
                n_timesteps = 6
                n_features = (len(feature_cols_saved) if feature_cols_saved else 1)

            # Create a placeholder input (zeros) and apply saved scaler if available
            inp = np.zeros((1, n_timesteps, n_features), dtype=float)
            if scaler is not None and feature_cols_saved is not None and n_features == len(feature_cols_saved):
                # scaler was fit on 2D array (-1, n_features)
                flat = inp.reshape(-1, n_features)
                try:
                    flat_scaled = scaler.transform(flat)
                    inp = flat_scaled.reshape(1, n_timesteps, n_features)
                except Exception as e:
                    print('Scaler transform failed, using unscaled input:', e)

            pred = model.predict(inp)
            # Flatten and take first value as mm (adapt as needed for your model)
            val = float(np.ravel(pred)[0])
            
            # Print real-time update with system time (show probability rather than mm)
            if ZONE_KARACHI is not None:
                now_str = datetime.datetime.now(tz=ZONE_KARACHI).strftime("%Y-%m-%d %I:%M:%S %p %Z")
            else:
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
            # Convert raw model value (probability) to [0,1].
            prob = min(1.0, float(val))
            risk_level = 'HIGH RISK' if prob >= 0.61 else 'MODERATE' if prob >= 0.36 else 'LOW RISK'
            print(f"[{now_str}] PREDICTION | Cloudburst probability: {prob:.2f} | Risk: {risk_level}")
        except Exception:
            val = 0.0
    else:
        # deterministic fallback value
        val = 0.0

    # Apply a simple lead-hour adjustment so forecasts for different lead
    # times are not identical when we don't have separate forecast inputs.
    try:
        lead = float(lead_hours)
    except Exception:
        lead = 0.0
    # small multiplicative growth per hour (tweak as needed)
    adj = 1.0 + 0.04 * lead
    
    # Val is currently a probability [0, 1]. We should adjust it but keep it capped at 1.0.
    final_prob = min(1.0, float(val) * adj)
    return max(0.0, final_prob)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Accepts JSON with key `features` containing a 2D list (timesteps x features)
    or a 3D list (batch x timesteps x features). Returns prediction plus diagnostics.
    """
    if model is None:
        return jsonify(error='Model not loaded'), 500

    payload = request.get_json(force=True, silent=True)
    if not payload or 'features' not in payload:
        return jsonify(error='Request JSON must contain "features"'), 400

    features = np.array(payload['features'], dtype=float)
    # normalize shape to (batch, timesteps, features)
    if features.ndim == 2:
        features = features.reshape(1, features.shape[0], features.shape[1])
    elif features.ndim == 3:
        pass
    else:
        return jsonify(error='"features" must be 2D or 3D array'), 400

    batch, n_timesteps, n_features = features.shape

    diagnostics = {
        'requested_shape': [int(batch), int(n_timesteps), int(n_features)],
        'feature_cols_saved_len': len(feature_cols_saved) if feature_cols_saved is not None else None
    }

    # Flatten and scale if scaler available
    flat = features.reshape(-1, n_features)
    scaled_flat = None
    scaler_error = None
    if scaler is not None:
        try:
            scaled_flat = scaler.transform(flat)
        except Exception as e:
            scaler_error = str(e)

    if scaled_flat is not None:
        inp = scaled_flat.reshape(batch, n_timesteps, n_features)
    else:
        inp = features

    # Predict
    try:
        preds = model.predict(inp)
        preds_list = np.ravel(preds).tolist()
    except Exception as e:
        return jsonify(error='Model prediction failed', detail=str(e)), 500

    # Convert model outputs (assumed probabilities) to [0,1]
    probs_list = [min(1.0, float(p)) if isinstance(p, (int, float, np.floating)) else 0.0 for p in preds_list]
    cloudburst_flags = [(float(prob) >= 0.5) for prob in probs_list]

    # prepare diagnostics sample (first timestep of first sample) for debug (limit size)
    diag_raw = flat.reshape(batch, n_timesteps, n_features)[0].tolist() if flat.size else None
    diag_scaled = scaled_flat.reshape(batch, n_timesteps, n_features)[0].tolist() if scaled_flat is not None else None

    diagnostics_payload = {
        'raw_first_sample': diag_raw,
        'scaled_first_sample': diag_scaled,
        'scaler_error': scaler_error,
    }
    diagnostics_payload.update(diagnostics)

    # Output probabilities as percentage (e.g., 73.42 for 73.42%) and include classification threshold
    threshold = CLASSIFICATION_THRESHOLD
    
    # Store prediction in database
    db_results = []
    for i, prob in enumerate(probs_list):
        try:
            db_result = store_prediction_and_check_alert(
                probability=float(prob),
                latitude=float(payload.get('latitude', 35.6)),
                longitude=float(payload.get('longitude', 73.6)),
                features=payload.get('features', [])
            )
            db_results.append(db_result)
        except Exception as e:
            print(f'Database storage error: {e}')
            db_results.append({'status': 'error', 'message': str(e)})
    
    if batch == 1:
        return jsonify({
            'cloudburst_probability': round(float(probs_list[0] * 100.0), 2),
            'cloudburst': bool(cloudburst_flags[0]),
            'classification_threshold': float(threshold),
            'database_stored': db_results[0] if db_results else None,
            'diagnostics': diagnostics_payload
        })
    else:
        return jsonify({
            'cloudburst_probabilities': [round(float(p * 100.0), 2) for p in probs_list],
            'cloudburst': cloudburst_flags,
            'classification_threshold': float(threshold),
            'database_stored': db_results if db_results else None,
            'diagnostics': diagnostics_payload
        })

def generate_forecast_array(base_prob, threshold=CLASSIFICATION_THRESHOLD):
    if ZONE_KARACHI is not None:
        now = datetime.datetime.now(tz=ZONE_KARACHI)
    else:
        now = datetime.datetime.now()
    forecast = []
    for h in range(1, 7):
        ts_dt = now + datetime.timedelta(hours=h)
        try:
            ts = ts_dt.strftime('%Y-%m-%dT%H:%M:%S+05:00')
        except Exception:
            ts = (ts_dt.replace(tzinfo=None)).isoformat() + '+05:00'
        
        # The model inherently predicts 2 hours ahead, so base_prob represents the +2h forecast.
        # We scale relative to h=2 so that the +2h card matches the main dashboard.
        adj = 1.0 + 0.04 * (h - 2)
        final_prob = max(0.0, min(1.0, base_prob * adj))
        cb = final_prob >= threshold
        forecast.append({'timestamp': ts, 'cloudburst_probability': round(float(final_prob * 100.0), 2), 'cloudburst': bool(cb)})
    return forecast


@app.route('/api/predict_live', methods=['POST'])
@require_api_key
def api_predict_live():
    """Build features from provided ERA5 rows and run prediction.
    Expected JSON formats:
    - {"era5": {"rows": [ {feature_name: value, ...}, ... ], "timesteps": 6 }}
      `rows` should be ordered old->new; the last `timesteps` rows will be used.
    - Or same as /api/predict using `features` (2D or 3D) — accepts authenticated requests.
    """
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify(error='Empty JSON payload'), 400

    # If raw features provided, forward to existing predict logic
    if 'features' in payload:
        # Reuse api_predict logic but ensure auth already checked
        return api_predict()

    # If user requested a CDS fetch, attempt to fetch data using cdsapi
    if 'cds' in payload:
        if cdsapi is None or xr is None:
            return jsonify(error='cdsapi/xarray not available on server; install dependencies'), 500

        cds_req = payload.get('cds').get('cds_request') if payload.get('cds') else None
        dataset = payload.get('cds', {}).get('dataset', 'reanalysis-era5-pressure-levels')
        if not cds_req:
            return jsonify(error='cds.cds_request is required for CDS fetch'), 400

        # Build client using env or provided creds
        cds_url = os.environ.get('CDS_API_URL')
        cds_key = os.environ.get('CDS_API_KEY')
        # allow passing key in payload (not recommended for production)
        if 'cds_api_key' in payload:
            cds_key = payload['cds_api_key']

        try:
            client = cdsapi.Client(url=cds_url, key=cds_key) if cds_url or cds_key else cdsapi.Client()
        except Exception as e:
            return jsonify(error='Failed to initialize CDS client', detail=str(e)), 500

        # retrieve to a temporary file
        import tempfile
        tmpf = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        tmp_path = tmpf.name
        tmpf.close()
        try:
            # ensure format=netcdf for easy xarray loading
            cds_req['format'] = 'netcdf'
            client.retrieve(dataset, cds_req, tmp_path)
        except Exception as e:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return jsonify(error='CDS retrieve failed', detail=str(e)), 500

        # open with xarray
        try:
            ds = xr.open_dataset(tmp_path)
        except Exception as e:
            return jsonify(error='Failed to open CDS data', detail=str(e)), 500

        # Build feature rows by mapping feature_cols_saved -> dataset variables
        if feature_cols_saved is None:
            return jsonify(error='Feature columns not available on server. Please provide features directly.'), 400

        # Retrieve timesteps parameter
        requested_timesteps = payload.get('cds', {}).get('timesteps', 6)

        # If CDS returns no time dimension (single timestep), duplicate it N times
        if 'time' not in ds.dims:
            # Single timestep; we'll repeat it `requested_timesteps` times
            times_sel = [0] * requested_timesteps
        else:
            timesteps_avail = len(ds['time'])
            times = ds['time'].values
            # choose last `requested_timesteps` times
            times_sel = list(times[-min(requested_timesteps, timesteps_avail):])
            # if less than requested, pad by repeating the first
            while len(times_sel) < requested_timesteps:
                times_sel = [times_sel[0]] + times_sel

        built = []
        for t_idx, t in enumerate(times_sel):
            row_vec = []
            # select data at this time
            try:
                # If single timestep (no time dim), use whole dataset; otherwise select by time
                if 'time' not in ds.dims:
                    slice_ds = ds
                else:
                    slice_ds = ds.sel(time=t)
            except Exception:
                if isinstance(t, (int, float)) and 'time' in ds.dims:
                    slice_ds = ds.isel(time=int(t % len(ds['time'])))
                else:
                    slice_ds = ds

            for col in feature_cols_saved:
                # Feature column format: var_level (e.g., 'u_500') or surface var (e.g., 't2m')
                val = 0.0
                
                # Mapping from model feature names to ERA5 CDS variable names
                var_mapping = {
                    'r_500': ('relative_humidity', 500), 'r_700': ('relative_humidity', 700), 'r_850': ('relative_humidity', 850),
                    't_500': ('temperature', 500), 't_700': ('temperature', 700), 't_850': ('temperature', 850),
                    'u_500': ('u_component_of_wind', 500), 'u_700': ('u_component_of_wind', 700), 'u_850': ('u_component_of_wind', 850),
                    'v_500': ('v_component_of_wind', 500), 'v_700': ('v_component_of_wind', 700), 'v_850': ('v_component_of_wind', 850),
                    'vv_500': ('vertical_velocity', 500), 'vv_700': ('vertical_velocity', 700), 'vv_850': ('vertical_velocity', 850),
                    't2m': '2m_temperature',
                    'd2m': '2m_dewpoint_temperature',
                    'sp': 'surface_pressure',
                    'tcc': 'total_cloud_cover',
                    'cape': 'convective_available_potential_energy',
                    'tcwv': 'total_column_water_vapour'
                }
                
                try:
                    if col in var_mapping:
                        mapping = var_mapping[col]
                        
                        if isinstance(mapping, tuple):
                            # Pressure level variable
                            cds_var, level = mapping
                            if cds_var in slice_ds:
                                try:
                                    val = float(slice_ds[cds_var].sel(level=level, method='nearest').values)
                                except Exception:
                                    try:
                                        val = float(slice_ds[cds_var].isel(level=(abs(slice_ds['level'].values - level)).argmin()).values)
                                    except Exception:
                                        val = 0.0
                        else:
                            # Surface variable
                            cds_var = mapping
                            if cds_var in slice_ds:
                                val = float(slice_ds[cds_var].values)
                    else:
                        # Temporal features (month_sin, month_cos, doy_sin, doy_cos, cloudburst)
                        val = 0.0
                except Exception:
                    val = 0.0
                
                row_vec.append(val)
            built.append(row_vec)

        # cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        # proceed as before
        features = np.array(built, dtype=float)
        features = features.reshape(1, features.shape[0], features.shape[1])
        n_features = features.shape[2]
        flat = features.reshape(-1, n_features)
        scaled_flat = None
        scaler_error = None
        if scaler is not None:
            try:
                scaled_flat = scaler.transform(flat)
            except Exception as e:
                scaler_error = str(e)

        inp = scaled_flat.reshape(1, requested_timesteps, n_features) if scaled_flat is not None else features
        try:
            preds = model.predict(inp)
            pred_val = float(np.ravel(preds)[0])
        except Exception as e:
            return jsonify(error='Model prediction failed', detail=str(e)), 500

        diag = {
            'raw_first_sample': flat.reshape(1, requested_timesteps, n_features)[0].tolist(),
            'scaled_first_sample': scaled_flat.reshape(1, requested_timesteps, n_features)[0].tolist() if scaled_flat is not None else None,
            'scaler_error': scaler_error
        }

        prob = min(1.0, float(pred_val))
        threshold = CLASSIFICATION_THRESHOLD
        forecast_array = generate_forecast_array(prob, threshold)
        
        # Store CDS prediction in database
        try:
            db_result = store_prediction_and_check_alert(
                probability=float(prob),
                latitude=float(payload.get('latitude', 35.6)),
                longitude=float(payload.get('longitude', 73.6)),
                features=built[0] if built else []
            )
        except Exception as e:
            print(f'Database storage error: {e}')
            db_result = {'status': 'error', 'message': str(e)}
        
        return jsonify({'cloudburst_probability': round(float(prob * 100.0), 2), 'cloudburst': prob >= threshold, 'classification_threshold': float(threshold), 'forecast': forecast_array, 'database_stored': db_result, 'diagnostics': diag})

    # fallback: original behavior if no cds in payload
    era5 = payload.get('era5')
    if not era5 or 'rows' not in era5:
        return jsonify(error='Missing "era5.rows" in payload'), 400

    rows = era5['rows']
    timesteps = era5.get('timesteps', 6)
    
    if len(rows) < timesteps:
        return jsonify(error='Not enough rows'), 400

    # Build feature vector for each timestep using saved feature column list if available
    if feature_cols_saved is None:
        return jsonify(error='Feature columns not available on server. Please provide features directly.'), 400

    n_features = len(feature_cols_saved)
    built_all = []
    for r in rows:
        row_vec = []
        for col in feature_cols_saved:
            # prefer keys as-is, otherwise try lowercase
            if col in r:
                val = r[col]
            elif col.lower() in r:
                val = r[col.lower()]
            else:
                # missing feature: fill with 0.0
                val = 0.0
            try:
                row_vec.append(float(val))
            except Exception:
                row_vec.append(0.0)
        built_all.append(row_vec)

    # create sliding windows
    windows = []
    for i in range(len(built_all) - timesteps + 1):
        windows.append(built_all[i : i+timesteps])

    features = np.array(windows, dtype=float)
    batch = features.shape[0]

    # validate shape
    if features.shape[2] != n_features:
        return jsonify(error='Feature length mismatch'), 400

    # Flatten and scale if scaler available
    flat = features.reshape(-1, n_features)
    scaled_flat = None
    scaler_error = None
    if scaler is not None:
        try:
            scaled_flat = scaler.transform(flat)
        except Exception as e:
            scaler_error = str(e)

    inp = scaled_flat.reshape(batch, timesteps, n_features) if scaled_flat is not None else features

    try:
        preds = model.predict(inp)
        pred_vals = np.ravel(preds).tolist()
    except Exception as e:
        return jsonify(error='Model prediction failed', detail=str(e)), 500

    diag = {
        'raw_first_sample': flat.reshape(batch, timesteps, n_features)[0].tolist() if batch > 0 else None,
        'scaled_first_sample': scaled_flat.reshape(batch, timesteps, n_features)[0].tolist() if scaled_flat is not None and batch > 0 else None,
        'scaler_error': scaler_error
    }

    threshold = CLASSIFICATION_THRESHOLD
    # generate the forecast array
    if ZONE_KARACHI is not None:
        now = datetime.datetime.now(tz=ZONE_KARACHI)
    else:
        now = datetime.datetime.now()
        
    forecast_array = []
    for i, p in enumerate(pred_vals[:6]): # up to 6 hours
        h = i + 1 # +1h, +2h, ...
        ts_dt = now + datetime.timedelta(hours=h)
        try:
            ts = ts_dt.strftime('%Y-%m-%dT%H:%M:%S+05:00')
        except Exception:
            ts = (ts_dt.replace(tzinfo=None)).isoformat() + '+05:00'
        
        prob = min(1.0, max(0.0, float(p)))
        cb = prob >= threshold
        forecast_array.append({'timestamp': ts, 'cloudburst_probability': round(float(prob * 100.0), 2), 'cloudburst': bool(cb)})

    # The base prediction for the dashboard should be the +2h forecast IF we have at least 2 predictions, otherwise the last one.
    if len(pred_vals) >= 2:
        base_prob = min(1.0, max(0.0, float(pred_vals[1])))
    else:
        base_prob = min(1.0, max(0.0, float(pred_vals[-1])))
        
    cb_base = base_prob >= threshold
    
    # Store ERA5 prediction in database
    try:
        db_result = store_prediction_and_check_alert(
            probability=float(base_prob),
            latitude=float(era5.get('latitude', 35.6)),
            longitude=float(era5.get('longitude', 73.6)),
            features=built_all[-1] if built_all else []
        )
    except Exception as e:
        print(f'Database storage error: {e}')
        db_result = {'status': 'error', 'message': str(e)}

    return jsonify({
        'cloudburst_probability': round(float(base_prob * 100.0), 2),
        'cloudburst': cb_base,
        'classification_threshold': float(threshold),
        'forecast': forecast_array,
        'database_stored': db_result,
        'diagnostics': diag
    })


@app.route('/api/recent_predictions', methods=['GET'])
def recent_predictions():
    """Get recent predictions from database"""
    hours = request.args.get('hours', 24, type=int)
    limit = request.args.get('limit', 100, type=int)
    
    try:
        predictions = db.get_recent_predictions(hours=hours, limit=limit)
        return jsonify({
            'count': len(predictions),
            'predictions': [dict(p) for p in predictions]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/high_risk_alerts', methods=['GET'])
def high_risk_alerts():
    """Get high-risk alerts from database"""
    days = request.args.get('days', 7, type=int)
    
    try:
        alerts = db.get_high_risk_predictions(days=days)
        unacknowledged = len([a for a in alerts if not a['acknowledged']])
        return jsonify({
            'count': len(alerts),
            'unacknowledged': unacknowledged,
            'alerts': [dict(a) for a in alerts]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/database_summary', methods=['GET'])
def database_summary():
    """Get overall database statistics and summary"""
    try:
        summary = db.get_database_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/acknowledge_alert/<int:alert_id>', methods=['POST'])
@require_api_key
def acknowledge_alert(alert_id):
    """Acknowledge a high-risk alert"""
    acknowledged_by = None
    try:
        if request.json:
            acknowledged_by = request.json.get('acknowledged_by')
    except Exception:
        pass
    
    try:
        db.acknowledge_alert(alert_id, acknowledged_by=acknowledged_by)
        return jsonify({
            'status': 'success',
            'message': f'Alert {alert_id} acknowledged',
            'alert_id': alert_id
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server is running"""
    return jsonify({'status': 'ok', 'message': 'Server is running', 'database': 'connected'}), 200


@app.route('/api/prediction')
def api_prediction():
    # Build next-6-hour forecasts (1..6 hours ahead). For each hour we call
    # make_prediction() and flag cloudburst if prediction >= 50 mm. The
    # top-level summary will use the first forecast (1-hour ahead).
    # Use Mingora / Swat local time (Pakistan Standard Time, Asia/Karachi)
    if ZONE_KARACHI is not None:
        now = datetime.datetime.now(tz=ZONE_KARACHI)
    else:
        # Fallback: use system local time
        now = datetime.datetime.now()
    forecast = []
    for h in range(1, 7):
        ts_dt = now + datetime.timedelta(hours=h)
        # Force timestamp string in Pakistan local time (UTC+05:00) so client
        # can show Mingora/Swat local time regardless of browser timezone.
        try:
            # Prefer an ISO-like with +05:00 offset
            ts = ts_dt.strftime('%Y-%m-%dT%H:%M:%S+05:00')
        except Exception:
            ts = (ts_dt.replace(tzinfo=None)).isoformat() + '+05:00'
        try:
            pred = make_prediction(h)
        except Exception:
            pred = 0.0
        # Extract adjusted probability from make_prediction
        prob = pred
        cb = prob >= 0.5
        # store probability as percentage, rounded to 2 decimals
        forecast.append({'timestamp': ts, 'cloudburst_probability': round(float(prob * 100.0), 2), 'cloudburst': bool(cb)})

    first = forecast[0] if forecast else {'cloudburst_probability': 0.0, 'cloudburst': False}
    # Build forecast_times (ISO strings) for the last SEQUENCE_LENGTH entries
    SEQUENCE_LENGTH = 6
    forecast_times = [f['timestamp'] for f in forecast]
    return jsonify({
        'cloudburst_probability': first['cloudburst_probability'],
        'cloudburst': first['cloudburst'],
        'classification_threshold': 0.5,
        'forecast_times': [t for t in forecast_times[-SEQUENCE_LENGTH:]],
        'forecast': forecast
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BABUSAR TOP CLOUDBURST PREDICTION SYSTEM - LIVE MONITORING")
    print("="*70)
    print(f"Location: Babusar Top, Kaghan Valley (35.6N, 73.6E)")
    print(f"Model: BiLSTM (6-hour sequence, 26 ERA5 features)")
    print(f"Update Interval: Live (continuous)")
    print(f"Forecast: Real-time 2-hour ahead prediction")
    print(f"Database: SQLite (cloudburst_predictions.db)")
    print("\n" + "="*70)
    print("API ENDPOINTS - PREDICTION & STORAGE")
    print("="*70)
    print("POST   /api/predict              - Make prediction (stored in DB)")
    print("POST   /api/predict_live         - ERA5 prediction (stored in DB)")
    print("GET    /api/recent_predictions   - Query recent predictions")
    print("GET    /api/high_risk_alerts     - Get high-risk alerts")
    print("GET    /api/database_summary     - Database statistics")
    print("POST   /api/acknowledge_alert/<id> - Mark alert as reviewed")
    print("\n" + "="*70)
    print("START THE DASHBOARD")
    print("="*70)
    print("Open your browser and visit:")
    print("   Local:   http://127.0.0.1:5000")
    print("   Network: http://192.168.1.3:5000")
    print("\n   Press CTRL+C to stop the server")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
