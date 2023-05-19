from datetime import datetime, timedelta
import pandas as pd
import requests
import json
import os
import csv
import mysql.connector

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator

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
    if os.path.exists('/home/airflow/data/data.csv'):
        # get data from API with start_date=today and limit=25000
        start_date = datetime.now().strftime('%Y-%m-%d')
        url = f'https://publicapi.traffy.in.th/share/teamchadchart/download?limit=25000&last_activity_start={start_date}'
        r = requests.get(url, allow_redirects=True)
    else:
        # Download data
        url = 'https://publicapi.traffy.in.th/dump-csv-chadchart/bangkok_traffy.csv'
        r = requests.get(url, allow_redirects=True)
    open('/home/airflow/data/data.csv', 'wb+').write(r.content)


def clean_data():
    """
    Load data from CSV to Pandas DataFrame, clean it, and save to Redshift
    """

    # load data from CSV to Pandas DataFrame
    df = pd.read_csv("/home/airflow/data/data.csv")
    df = df.dropna()
    # split cords column into latitude and longitude
    df[['longitude', 'latitude']] = df.coords.str.split(",", expand=True)

    # write file to CSV
    df.to_csv("/home/airflow/data/data_clean.csv", index=False)


def insert_update_table_from_csv():
    # Connect to the MySQL database
    conn = mysql.connector.connect(
        host="mysql", database="traffy", user="traffy", password="fondue", use_unicode=True, charset="utf8mb4")
    cursor = conn.cursor()

    # Enforce UTF-8 for the connection.
    cursor.execute('SET NAMES utf8mb4')
    cursor.execute("SET CHARACTER SET utf8mb4")
    cursor.execute("SET character_set_connection=utf8mb4")

    # Check if the table already exists
    cursor.execute("SHOW TABLES LIKE %s", ("traffy",))
    table_exists = cursor.fetchone()
    temp_data = []
    # Create the table if it doesn't exist
    if not table_exists:
        columns = """
            ticket_id VARCHAR(255) NOT NULL,
            type VARCHAR(255),
            organization VARCHAR(1000),
            comment VARCHAR(10000),
            coords VARCHAR(255),
            photo VARCHAR(255),
            photo_after VARCHAR(255),
            address VARCHAR(1000),
            subdistrict VARCHAR(255),
            district VARCHAR(255),
            province VARCHAR(255),
            timestamp VARCHAR(255),
            state VARCHAR(255),
            star VARCHAR(255),
            count_reopen VARCHAR(255),
            last_activity VARCHAR(255),
            latitude VARCHAR(255),
            longitude VARCHAR(255)
            """
        create_table_query = f"CREATE TABLE traffy ({columns})"
        cursor.execute(create_table_query)
        alt_query1 = f"ALTER TABLE traffy ADD PRIMARY KEY (ticket_id)"
        alt_query2 = "ALTER TABLE traffy CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        cursor.execute(alt_query1)
        cursor.execute(alt_query2)

    def do_insert():
        insert_query = f"""
        INSERT INTO traffy
            ({', '.join(header)})
        VALUES
            ({placeholders})
        ON DUPLICATE KEY UPDATE
            ticket_id = VALUES(ticket_id),
            type = VALUES(type),
            organization = VALUES(organization),
            comment = VALUES(comment),
            coords = VALUES(coords),
            photo = VALUES(photo),
            photo_after = VALUES(photo_after),
            address = VALUES(address),
            subdistrict = VALUES(subdistrict),
            district = VALUES(district),
            province = VALUES(province),
            timestamp = VALUES(timestamp),
            state = VALUES(state),
            star = VALUES(star),
            count_reopen = VALUES(count_reopen),
            last_activity = VALUES(last_activity),
            latitude = VALUES(latitude),
            longitude = VALUES(longitude)
        """
        cursor.executemany(insert_query, temp_data)
        print(f"Inserted {len(temp_data)} rows")
        conn.commit()

    # Read the CSV file and insert/update rows
    with open("/home/airflow/data/data_clean.csv", 'r') as file:
        print("Reading CSV file")
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        placeholders = ', '.join(['%s'] * len(header))

        for row in csv_reader:
            temp_data.append(tuple(row))
            if len(temp_data) > 999:
                do_insert()
                temp_data = []

    # Commit the changes and close the connectionif temp_data:
    if temp_data:
        do_insert()

    cursor.close()
    conn.close()


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

    insert_update_data_task = PythonOperator(
        task_id='insert_update_data',
        python_callable=insert_update_table_from_csv,
        dag=dag
    )

    # define task dependencies
    get_data_task >> clean_data_task >> insert_update_data_task
