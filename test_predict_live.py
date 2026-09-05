import urllib.request
import json
import datetime

# match the exact structure `fetchOpenMeteoFeatures` produces
payload = {
    "era5": {
        "rows": [
            {
                "r_500": 50.0, "r_700": 60.0, "r_850": 70.0,
                "t_500": 260.0, "t_700": 270.0, "t_850": 280.0,
                "u_500": 5.0, "u_700": 5.0, "u_850": 5.0,
                "v_500": 1.0, "v_700": 1.0, "v_850": 1.0,
                "vv_500": 0.0, "vv_700": 0.0, "vv_850": 0.0,
                "t2m": 290.0, "d2m": 280.0, "sp": 101300.0,
                "tcc": 0.5, "cape": 0.0, "tcwv": 0.0
            }
        ] * 6, # 6 identical timesteps for speed
        "timesteps": 6
    }
}

req = urllib.request.Request(
    'http://127.0.0.1:5000/api/predict_live',
    data=json.dumps(payload).encode('utf-8'),
    headers={
        'Content-Type': 'application/json',
        'X-API-Key': 'devkey'
    }
)

try:
    with urllib.request.urlopen(req) as response:
        print("Success:", response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
except Exception as e:
    print("Error:", e)
