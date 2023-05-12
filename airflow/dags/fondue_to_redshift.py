import os
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd
import requests

# For PythonOperator
def get_data_from_request():
    # Check if data exists
    if os.path.exists('../data/current.csv'):
        # Download data
        url = 'https://publicapi.traffy.in.th/share/teamchadchart/download'
        r = requests.get(url, allow_redirects=True)
        open('../data/current.csv', 'wb').write(r.content)
        # Create last update
        last_update = pd.DataFrame({'last_update': [pd.Timestamp.now()]})
        last_update.to_csv('../data/last_update.csv', index=False)
    else:
        url = 'https://publicapi.traffy.in.th/share/teamchadchart/search'
        # Read last update
        last_update = pd.read_csv('../data/last_update.csv')
        params = {
            'limit': 25000,
            'start_date': pd.Timestamp(last_update['last_update'][0]).strftime('%Y-%m-%d'),
            'end_date': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
        r = requests.get(url, params=params, allow_redirects=True)
        open('../data/current.csv', 'wb').write(r.content)
        # Update last update
        last_update = pd.DataFrame({'last_update': [pd.Timestamp.now()]})
        last_update.to_csv('../data/last_update.csv', index=False)


def clean_data():
    



default_args = {
    'owner': 'datath',
    'depends_on_past': False,
    'catchup': False,
    'start_date': days_ago(0),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'traffy_fondue_to_Redshift',
    default_args=default_args,
    description='Pipeline for getting data from Traffy Fondue and inserting it to Redshift',
    schedule_interval=timedelta(days=1),
)

t1 = PythonOperator(
    task_id="get_data_from_request",
    python_callable=get_data_from_request,
    dag=dag,
)

t2 = PythonOperator(
    task_id="clean_data",
    python_callable=clean_data,
    dag=dag,
)

t3 = PythonOperator(
    task_id="insert_data_to_db",
    python_callable=insert_data_to_db,
    dag=dag,
)

t1 >> t2
