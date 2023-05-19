from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
import os
import mysql.connector

# Retrieve the connection details from environment variables
# host = os.getenv("DB_HOST")
# port = os.getenv("DB_PORT")
# database = os.getenv("DB_NAME")
# user = os.getenv("DB_USER")
# password = os.getenv("DB_PASSWORD")
host = "127.0.0.1"
port = "3308"
database = "traffy"
user = "traffy"
password = "fondue"
def clean_data():
    # Establish a connection to the PostgreSQL database
    print(host, " ", port, " ", database, " ", user, " ", password)
    conn = mysql.connector.connect( user="traffy", password="fondue", host="localhost", port="3308", database="traffy",auth_plugin='mysql_native_password', use_unicode=True, charset="utf8mb4")

    sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM traffy
                               ''', conn)
    
    # Load data
    Traffyticket = pd.DataFrame(sql_query, columns=['ticket_id', 'type', 'organization', 'comment', 'photo', 'photo_after', 'address', 'subdistrict', 'district', 'province', 'timestamp', 'state', 'star', 'count_reopen', 'last_activity', 'latitude', 'longitude' ])
    # Traffyticket = pd.read_csv(data_path)

    print(f"Number of rows: {Traffyticket.shape[0]}")
    # Filter to collect ticket which state is 'เสร็จสิ้น'
    Traffyticket = Traffyticket[Traffyticket['state'] == 'เสร็จสิ้น']
    Traffyticket.head(3)

    # Filter to remove nan value
    print("Dropping nan value....")
    Traffyticket = Traffyticket.dropna()    
    print(f"Number of rows: {Traffyticket.shape[0]}")

    drop_columns = ['photo', 'photo_after', 'ticket_id', 'address', 'comment', 'state', 'last_activity', 'timestamp', 'star', 'subdistrict', 'organization', 'count_reopen', 'latitude', 'longitude' ]
    # drop_columns = ['photo', 'photo_after', 'ticket_id', 'coords', 'address', 'comment', 'state', 'last_activity', 'timestamp', 'star', 'subdistrict', 'organization', 'count_reopen']

    # Calculate by convert last_activity and timestamp to datetime and calculate to add the duration column
    Traffyticket['last_activity'] = pd.to_datetime(Traffyticket['last_activity'])
    Traffyticket['timestamp'] = pd.to_datetime(Traffyticket['timestamp'])

    # Convert duration to hours
    Traffyticket['duration'] = (Traffyticket['last_activity'] - Traffyticket['timestamp']).dt.total_seconds() / 3600

    # Show the result
    # Traffyticket.head(3)

    # Drop the columns
    Traffyticket.drop(drop_columns, axis=1, inplace=True)
    # Reset index
    Traffyticket.reset_index(drop=True, inplace=True)
    # Show the result
    Traffyticket.head(3)

    def extract_types(df):
        result = []
        for index,row in Traffyticket.iterrows():
            types = row['type'].strip('{}').split(',')
            for t in types: 
                new_row = row.copy()
                new_row['type'] = t.strip()
                result.append(new_row)
        return result


    Traffyticket = pd.DataFrame(extract_types(Traffyticket))

    # Traffyticket.head(10)

    # ## Prepare the data before train the model
    enc = OneHotEncoder(handle_unknown='ignore')

    nominal_columns = ['type', 'district', 'province']

    # Fit the encoder
    enc.fit(Traffyticket[nominal_columns])

    # Transform the categorical columns to numerical columns
    enc_cols = enc.transform(Traffyticket[nominal_columns]).toarray()

    # Create the new datafram with the encoded columns
    enc_df = pd.DataFrame(enc_cols, columns=enc.get_feature_names_out(nominal_columns))

    # Merge the original dataframe with the encoded dataframe
    Traffyticket = enc_df.join(Traffyticket)

    # Drop the original categorical columns 
    Traffyticket.drop(nominal_columns, axis=1, inplace=True)

    print(f"Shape of the dataframe before dropping: {Traffyticket.shape}")

    # Drop NaN value
    Traffyticket.dropna(inplace=True)

    print(f"Shape of the dataframe after dropping: {Traffyticket.shape}")
    # Show the result
    Traffyticket.head(10)

    print("Already drop NaN value and use oneHotEncoder.")
    # print(Traffyticket.info())

    # Close the connection
    conn.close()
    return Traffyticket