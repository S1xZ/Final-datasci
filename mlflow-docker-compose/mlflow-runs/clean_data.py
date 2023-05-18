from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


def clean_data(data_path):
    # Load data
    Traffyticket = pd.read_csv(data_path)

    print(f"Number of rows: {Traffyticket.shape[0]}")
    # Filter to collect ticket which state is 'เสร็จสิ้น'
    Traffyticket = Traffyticket[Traffyticket['state'] == 'เสร็จสิ้น']
    Traffyticket.head(3)

    # Filter to remove nan value
    print("Dropping nan value....")
    Traffyticket = Traffyticket.dropna()    
    print(f"Number of rows: {Traffyticket.shape[0]}")

    drop_columns = ['photo', 'photo_after', 'ticket_id', 'coords', 'address', 'comment', 'state', 'last_activity', 'timestamp', 'star', 'subdistrict', 'organization', 'count_reopen']

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

    return Traffyticket