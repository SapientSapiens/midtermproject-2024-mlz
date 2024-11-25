# importing the basic  libraries
import pandas as pd
import numpy as np
import requests

cloudURL = 'midterm-2024-cloud.eba-vptaqmf4.eu-north-1.elasticbeanstalk.com'
url_predict = f'http://{cloudURL}/predict'

print(f'cloud URL is {url_predict}')

# raw patient-appointment record from the test dataset
appointment = {
    'PatientId' : 5.288218e+13,
    'AppointmentID' : 5665685,
    'Gender': 'F',
    'ScheduledDay': '2016-05-05T15:04:36Z',
    'AppointmentDay': '2016-05-30T00:00:00Z',
    'Age': 58,
    'Neighbourhood': 'SANTOS DUMONT',
    'Scholarship': 0,
    'Hipertension': 1,
    'Diabetes': 1,
    'Alcoholism': 0,
    'Handcap': 0,
    'SMS_received': 1,
    'No-show': 'Yes'
}

# since the model has been trained on the cleaned, prepared data and with feature engineering,
# this raw data from the test set cannot be directly applied to the model. 
# also, historical data of the patient (previous & missed appointments) which is part of the
# feature engineering cannot be determined just from the single record.
# So, we shall use the cleaned_prepared_df.pkl file generated from the train.py script.

# loading the cleaned, prepared and featured engineered dataframe from the pickle file
df_model = pd.read_pickle('cleaned_prepared_df.pkl')

# dropping the outcome/target feature from the dataframe
del df_model['no_show']

# extracting the model suitable data with patient history from this dataframe
df_appointment= df_model[df_model['appointmentid']== appointment['AppointmentID']]

# all necessary information from 'scheduledday', 'appointmentday' were already extracted and feature engineered
del df_appointment['scheduledday']
del df_appointment['appointmentday']

# I choose F as 1 because number females are almost twice that of males and so more 1s & lesser 0s would be generated.
#Lesser zeros would prevent the increase of sparse-ness in the dataset
#df_appointment.gender = (df_appointment.gender == 'F').astype(int)
df_appointment.loc[:, 'gender'] = (df_appointment['gender'] == 'F').astype(int)


# converting dataframe to dictionary
df_appointment_dict = df_appointment.to_dict(orient='records')[0]

# handling propable exceptions
try:
    response = requests.post(url_predict, json=df_appointment_dict, timeout=5).json()  # Added timeout for faster failure
    percent_no_show = response["no_show_probability"]*100

    if response['is_no_show'] == True:
        print(f'The patient would probably NOT SHOW UP for the appointment number {df_appointment_dict["appointmentid"]} with a {round(percent_no_show, 2)} percent possibility')
    else:
        print(f'The patient would probably SHOW UP for the appointment number {df_appointment_dict["appointmentid"]}')

except requests.exceptions.ConnectionError as e:
    print(f"Error: Could not connect to the prediction service at {url_predict}. Please ensure it is running.")
    print(f"Original error: {e}")