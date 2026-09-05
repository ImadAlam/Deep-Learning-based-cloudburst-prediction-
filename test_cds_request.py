import urllib.request, json
payload = {
  'cds': {
    'cds_request': {
      'variable': ['t'],
      'pressure_level': ['500'],
      'product_type': 'reanalysis',
      'year': '2023',
      'month': '01',
      'day': '01',
      'time': '00:00',
      'area': [37, 73, 36, 74]
    },
    'dataset': 'reanalysis-era5-pressure-levels',
    'timesteps': 1
  },
  'cds_api_key': '26372d57-2be4-4937-ac92-6f684a552dd3',
  'cds_api_url': 'https://cds.climate.copernicus.eu/api'
}
req = urllib.request.Request('http://127.0.0.1:5000/api/predict_live', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json','X-API-Key':'devkey'})
try:
    resp = urllib.request.urlopen(req, timeout=600)
    print("SUCCESS:", resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}:")
    try:
        error_body = e.read().decode()
        print(error_body)
    except:
        print(e)
except Exception as e:
    print(f"Error: {e}")
