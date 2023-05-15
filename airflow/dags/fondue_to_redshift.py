from datetime import datetime, timedelta
import pandas as pd
import requests
import json
import os

from airflow import DAG
from airflow.hooks.postgres_hook import PostgresHook
from airflow.operators.python_operator import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    # set start time to 23.59 of everyday
    'start_date': datetime(2023, 5, 14, 23, 59),
    'retries': 0,
}

dag = DAG(
    'teamchadchart_dag',
    default_args=default_args,
    catchup=False,
    schedule_interval='@daily',
)


def get_data():
    """
    Fetch data from API and save as CSV
    """
    # check if data.csv exists
    if os.path.exists('./data/data.csv'):
        # get data from API with start_date=today and limit=25000
        start_date = datetime.now().strftime('%Y-%m-%d')
        url = f'https://publicapi.traffy.in.th/share/teamchadchart/download?limit=25000&start_date={start_date}'
        r = requests.get(url, allow_redirects=True)
    else:
        # Download data
        url = 'https://publicapi.traffy.in.th/dump-csv-chadchart/bangkok_traffy.csv'
        r = requests.get(url, allow_redirects=True)

    open('./data/data.csv', 'wb').write(r.content)


def clean_data():
    """
    Load data from CSV to Pandas DataFrame, clean it, and save to Redshift
    """

    redshift_hook = PostgresHook(postgres_conn_id='redshift')

    # load data from CSV to Pandas DataFrame
    df = pd.read_csv("../data/data.csv")

    # drop empty fields
    df = df.dropna()

    # split cords column into latitude and longitude
    df[['latitude', 'longitude']] = pd.DataFrame(
        df['cords'].tolist(), index=df.index)

    # save cleaned data back to Redshift
    df.to_sql(
        'teamchadchart',
        redshift_hook.get_engine(),
        if_exists='replace',
        index=False
    )


with dag:
    # define tasks
    get_data_task = PythonOperator(
        task_id='get_data',
        python_callable=get_data,
        dag=dag
    )

    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        dag=dag
    )

    # define task dependencies
    get_data_task >> clean_data_task
