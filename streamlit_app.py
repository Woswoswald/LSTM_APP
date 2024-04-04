from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import joblib
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import bcrypt
import random
import streamlit as st
import pandas as pd
import requests

# Connect to the SQLite database for user credentials
conn_users = sqlite3.connect('users.db')
c_users = conn_users.cursor()

# Create a table to store user credentials
c_users.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT)''')

# Function to insert user credentials into the database
def insert_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    c_users.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn_users.commit()

# Function to check user credentials (hardcoded for a single user)
def authenticate_user(username, password):
    # Hardcoded username and password
    hardcoded_username = "SlavaOsw"
    hardcoded_password = "334455"

    # Check if the provided credentials match the hardcoded credentials
    if username == hardcoded_username and password == hardcoded_password:
        return True
    else:
        return False


# Connect to the SQLite database for predictions
conn_predictions = sqlite3.connect('predictions.db')
c_predictions = conn_predictions.cursor()

# Create a table to store predictions
c_predictions.execute('''CREATE TABLE IF NOT EXISTS predictions
             (date_range TEXT, predictions TEXT, real_values TEXT)''')

# Function to insert predictions into the database
def insert_predictions(date_range, predictions, real_values):
    predictions_str = ', '.join([str(round(pred)) for pred in predictions])
    real_values_str = ', '.join([str(value) for value in real_values])
    c_predictions.execute("INSERT INTO predictions (date_range, predictions, real_values) VALUES (?, ?, ?)", (date_range, predictions_str, real_values_str))
    conn_predictions.commit()

# Function to fetch data from the predictions database
def get_predictions():
    c_predictions.execute("SELECT * FROM predictions")
    return c_predictions.fetchall()

# Streamlit UI for main application
def main():
    st.title("Login to Access the Application")

    # Login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login_button = st.button("Login")

    if authenticate_user(username, password):
        st.success("Login successful!")
        show_application()
    elif login_button:
        st.error("Invalid username or password")

def show_application():
    st.title("Prediction App")

    # Button to clear the database
    if st.button("Clear Database"):
        # Clear all rows from the predictions table
        c_predictions.execute("DELETE FROM predictions")
        conn_predictions.commit()
        st.success("Database cleared successfully!")

    # Input for start and end dates using date picker
    start_date_input = st.date_input("Select the start date:")

    # Automatically set the end date to the Sunday of the same week as the chosen Monday
    end_date_input = start_date_input + timedelta(days=(6 - start_date_input.weekday()))

    # Display the selected dates
    st.write(f"Selected Start Date: {start_date_input.strftime('%Y-%m-%d')}")
    st.write(f"Automatically Set End Date: {end_date_input.strftime('%Y-%m-%d')}")

    # Check if the selected start date is not Monday
    if start_date_input.weekday() != 0:  # Monday corresponds to 0
        st.error("Please choose Monday as the first input date.")
    else:
        # Button to trigger the analysis
        if st.button("Generate Sequences"):
            # Generate random real values
            real_values = [random.randint(1100, 1500) for _ in range(7)]

            # Create columns for better layout
            col1, col2, col3, col4 = st.columns(4)

            # Call the function for each indicator and display the sequences
            with col1:
                st.subheader("Sequence for Indicator 'Diarrhée':")
                sequence_6 = get_sequence_for_indicator(6, start_date_input, end_date_input)
                st.write(sequence_6)

            with col2:
                st.subheader("Sequence for Indicator 'Varicelle':")
                sequence_7 = get_sequence_for_indicator(7, start_date_input, end_date_input)
                st.write(sequence_7)

            with col3:
                st.subheader("Sequence for Indicator 'Syndromes Grippaux':")
                sequence_3 = get_sequence_for_indicator(3, start_date_input, end_date_input)
                st.write(sequence_3)

            with col4:
                st.subheader("Sequence for Temp. max (Weather):")
                sequence_weather = get_sequence_for_weather(start_date_input, end_date_input)
                st.write(sequence_weather)

            # Display sequence for HOS_PRES_TOTAL
            st.subheader("Sequence for HOS_PRES_TOTAL:")
            sequence_hos_pres_total = get_sequence_for_HOS_PRES_TOTAL(start_date_input, end_date_input)
            st.write(sequence_hos_pres_total)

                    #Importing scalers
            scaler_dir = "scalers/"

            with open(scaler_dir + 'scaler_diarrhee.pkl', 'rb') as f:
                scaler_dia = joblib.load(f)

            with open(scaler_dir + 'scaler_grippe.pkl', 'rb') as f:
                scaler_grippe = joblib.load(f)

            with open(scaler_dir + 'scaler_varicelle.pkl', 'rb') as f:
                scaler_varicelle = joblib.load(f)

            with open(scaler_dir + 'scaler_valeur.pkl', 'rb') as f:
                scaler_valeur = joblib.load(f)

            with open(scaler_dir + 'scaler_temp_max.pkl', 'rb') as f:
                scaler_temp = joblib.load(f)

            model_1 = keras.models.load_model("models/hos_pres_total_1.h5")
            model_2 = keras.models.load_model("models/hos_pres_total_2.h5")
            model_3 = keras.models.load_model("models/hos_pres_total_3.h5")
            model_4 = keras.models.load_model("models/hos_pres_total_4.h5")
            model_5 = keras.models.load_model("models/hos_pres_total_5.h5")
            model_6 = keras.models.load_model("models/hos_pres_total_6.h5")
            model_7 = keras.models.load_model("models/hos_pres_total_7.h5")

            days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            def create_day_dummy(day):
                return [int(day == d) for d in days_of_week]

        # Test sequences
            test_seq_valeur = np.array(sequence_hos_pres_total)
            grippe_seq = np.array(sequence_3)
            varicelle_seq = np.array(sequence_7) 
            diarrhee_seq = np.array(sequence_6)
            temp_seq = np.array(sequence_weather)

            scaled_test_seq_valeur = scaler_valeur.transform(test_seq_valeur.reshape(-1, 1)).flatten()
            scaled_grippe_seq = scaler_grippe.transform(grippe_seq.reshape(-1, 1)).flatten()
            scaled_varicelle_seq = scaler_varicelle.transform(varicelle_seq.reshape(-1, 1)).flatten()
            scaled_diarrhee_seq = scaler_dia.transform(diarrhee_seq.reshape(-1, 1)).flatten()
            scaled_temp_seq = scaler_temp.transform(temp_seq.reshape(-1, 1)).flatten()

            input_seq = []

            for i in range(len(scaled_test_seq_valeur)):
                day_of_week = days_of_week[i % 7]
                day_dummy = create_day_dummy(day_of_week)

                input_row = np.concatenate(([scaled_test_seq_valeur[i], scaled_varicelle_seq[i], scaled_grippe_seq[i], scaled_diarrhee_seq[i], scaled_temp_seq[i]], day_dummy))
                input_seq.append(input_row)

            input_seq = np.array(input_seq)

            st.subheader("Input Sequence:")
            st.table(pd.DataFrame(input_seq, columns=[
                'Scaled Test Sequence (Valeur)',
                'Scaled Grippe Sequence',
                'Scaled Varicelle Sequence',
                'Scaled Diarrhee Sequence',
                'Scaled Temp Sequence',
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ]))

            j1 = model_1.predict(input_seq.reshape(1, 7, 12))
            j2 = model_2.predict(input_seq.reshape(1, 7, 12))
            j3 = model_3.predict(input_seq.reshape(1, 7, 12))
            j4 = model_4.predict(input_seq.reshape(1, 7, 12))
            j5 = model_5.predict(input_seq.reshape(1, 7, 12))
            j6 = model_6.predict(input_seq.reshape(1, 7, 12))
            j7 = model_7.predict(input_seq.reshape(1, 7, 12))

            J1_orig = scaler_valeur.inverse_transform(j1)
            J2_orig = scaler_valeur.inverse_transform(j2)
            J3_orig = scaler_valeur.inverse_transform(j3)
            J4_orig = scaler_valeur.inverse_transform(j4)
            J5_orig = scaler_valeur.inverse_transform(j5)
            J6_orig = scaler_valeur.inverse_transform(j6)
            J7_orig = scaler_valeur.inverse_transform(j7)

        # Generate and display the plot of predicted values (j1 to j6)
            st.subheader("Predicted Values:")
            days_range = pd.date_range(start=start_date_input, periods=7)

            predictions = [J1_orig, J2_orig, J3_orig, J4_orig, J5_orig, J6_orig, J7_orig]

            model_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']

            predictions = [pred[0][0] for pred in predictions]

            plt.plot(predictions, label='Predicted Values')

            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()

            st.pyplot(plt)

            insert_predictions(f"{start_date_input.strftime('%Y-%m-%d')} to {end_date_input.strftime('%Y-%m-%d')}", predictions, real_values)

# Fetch data from the database and display it
    st.subheader("Predictions Database:")

    # Fetch data from the predictions database
    data = get_predictions()

    # Display data in a table format
    if data:
        df = pd.DataFrame(data, columns=['Date Range', 'Predictions for the next week', 'Real Values of the last week'])
        st.write(df)
    else:
        st.write("No data available in the predictions database.")

# Function to create a day dummy for neural network input
def create_day_dummy(day):
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return [int(day == d) for d in days_of_week]

def get_sequence_for_indicator(indicator, start_date_input, end_date_input):
    url = f"https://www.sentiweb.fr/api/v1/datasets/rest/incidence?indicator={indicator}&geo=REG&span=all"
    response = requests.get(url)
    data = response.json()

    # Filter data for Basse-Normandie
    filtered_data = [entry for entry in data["data"] if entry["geo_name"] == "BASSE-NORMANDIE"]

    # Function to convert week and year to date (Thursday to Wednesday)
    def week_to_date(week, year):
        start_date = datetime.strptime(f'{year}-W{week}-4', "%Y-W%W-%w").date()
        end_date = start_date + timedelta(days=6)
        return start_date, end_date

    for entry in filtered_data:
        week = entry['week']
        year = int(str(week)[:4])
        start_date, end_date = week_to_date(week % 100, year)
        entry['dates'] = {'start_date': start_date.strftime('%Y-%m-%d'), 'end_date': end_date.strftime('%Y-%m-%d')}

    try:
        start_date_str = start_date_input.strftime("%Y-%m-%d")
        end_date_str = end_date_input.strftime("%Y-%m-%d")

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

        second_inc100 = None
        last_inc100 = None

        for entry in filtered_data:
            entry_start_date = datetime.strptime(entry['dates']['start_date'], "%Y-%m-%d").date()
            entry_end_date = datetime.strptime(entry['dates']['end_date'], "%Y-%m-%d").date()

            if start_date <= entry_end_date and end_date >= entry_start_date:
                if second_inc100 is None:
                    second_inc100 = entry['inc100']
                last_inc100 = entry['inc100']

        if second_inc100 is not None and last_inc100 is not None:
            sequence = [last_inc100] * 3 + [second_inc100] * 4
            return sequence
        else:
            return f"No valid entries found in the specified date range for indicator {indicator}."

    except ValueError:
        return f"Invalid date format. Please enter dates in the format YYYY-MM-DD for indicator {indicator}."
    
def get_sequence_for_HOS_PRES_TOTAL(start_date_input, end_date_input):
    # Load the data from HOS_PRES_TOTAL_NEW.csv
    hos_pres_total_data = pd.read_csv("HOS_PRES_TOTAL_NEW.csv")

    # Convert DateCreateIndicHopTension to datetime
    hos_pres_total_data['DateCreateIndicHopTension'] = pd.to_datetime(hos_pres_total_data['DateCreateIndicHopTension']).dt.date

    try:
        start_date_str = start_date_input.strftime("%Y-%m-%d")
        end_date_str = end_date_input.strftime("%Y-%m-%d")

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

        # Filter data for the specified date range
        filtered_data = hos_pres_total_data[(hos_pres_total_data['DateCreateIndicHopTension'] >= start_date) &
                                            (hos_pres_total_data['DateCreateIndicHopTension'] <= end_date)]

        # Get the sequence of 7 corresponding values from ValeurIndicHopTension column
        sequence = filtered_data['ValeurIndicHopTension'].tolist()[:7]

        return sequence

    except ValueError:
        return "Invalid date format. Please enter dates in the format YYYY-MM-DD for HOS_PRES_TOTAL."
    
def get_sequence_for_weather(start_date_input, end_date_input):
    # Code for retrieving weather data
    url = f"https://www.meteo.bzh/climatologie/station/CAEN-CARPIQUET/mois/{start_date_input.strftime('%Y-%m')}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')

        if table:
            data = []
            for row in table.find_all('tr'):
                columns = row.find_all(['th', 'td'])
                row_data = [column.text.strip() for column in columns]
                data.append(row_data)

            columns = data[0]
            weather_data = pd.DataFrame(data[1:], columns=columns)

            # Find the index of the start date in the 'Jour' column
            start_date_str = start_date_input.strftime('%d/%m')
            start_date_index = weather_data[weather_data['Jour'] == start_date_str].index

            if not start_date_index.empty:
                start_date_index = start_date_index[0]

                # Get the sequence of 7 'Temp. max' values starting from the found index
                temp_max_sequence = weather_data.iloc[start_date_index:start_date_index + 7]['Temp. max'].str.replace('°C', '').astype(float).tolist()

                # If the sequence has less than 7 values, get the remaining values from the next month
                if len(temp_max_sequence) < 7:
                    next_month_url = f"https://www.meteo.bzh/climatologie/station/CAEN-CARPIQUET/mois/{(start_date_input + pd.DateOffset(months=1)).strftime('%Y-%m')}"
                    next_month_response = requests.get(next_month_url)

                    if next_month_response.status_code == 200:
                        next_month_soup = BeautifulSoup(next_month_response.text, 'html.parser')
                        next_month_table = next_month_soup.find('table')

                        if next_month_table:
                            next_month_data = []
                            for row in next_month_table.find_all('tr'):
                                columns = row.find_all(['th', 'td'])
                                row_data = [column.text.strip() for column in columns]
                                next_month_data.append(row_data)

                            next_month_columns = next_month_data[0]
                            next_month_weather_data = pd.DataFrame(next_month_data[1:], columns=next_month_columns)

                            # Get the remaining 'Temp. max' values from the next month
                            remaining_values = 7 - len(temp_max_sequence)
                            next_month_values = next_month_weather_data.iloc[:remaining_values]['Temp. max'].str.replace('°C', '').astype(float).tolist()

                            temp_max_sequence += next_month_values

                        else:
                            return "No table found on the page for the next month."

                    else:
                        return f"Failed to retrieve the webpage for the next month. Status code: {next_month_response.status_code}"

                return temp_max_sequence

            else:
                return f"Start date {start_date_str} not found in the weather data."

        else:
            return "No table found on the page."

    else:
        return f"Failed to retrieve the webpage. Status code: {response.status_code}"

if __name__ == "__main__":
    main()
