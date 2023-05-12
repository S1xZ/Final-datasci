import requests
import pandas as pd
import os

# For PythonOperator


def get_data_from_request():
    # Check if data exists
    if not os.path.exists('./data/current.csv'):
        # Download data
        url = 'https://publicapi.traffy.in.th/dump-csv-chadchart/bangkok_traffy.csv'
        r = requests.get(url, allow_redirects=True)
        open('./data/current.csv', 'wb').write(r.content)
        # Create last update
        last_update = pd.DataFrame({'last_update': [pd.Timestamp.now()]})
        last_update.to_csv('./data/last_update.csv', index=False)
    else:
        url = 'https://publicapi.traffy.in.th/share/teamchadchart/download'
        # Read last update
        last_update = pd.read_csv('./data/last_update.csv')
        params = {
            'limit': 25000,
            'last_activity_start': pd.Timestamp(last_update['last_update'][0]).strftime('%Y-%m-%d'),
            'last_activity_end': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
        r = requests.get(url, params=params, allow_redirects=True)
        open('./data/current.csv', 'wb').write(r.content)
        # Update last update
        last_update = pd.DataFrame({'last_update': [pd.Timestamp.now()]})
        last_update.to_csv('./data/last_update.csv', index=False)


get_data_from_request()
