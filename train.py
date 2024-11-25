# importing the basic  libraries
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from xgboost import DMatrix, train

#loading the dataset into a dataframe
df = pd.read_csv('KaggleV2-May-2016.csv')

# data cleaning, preparation and feature engineering

# Make column values uniform in categorical columns
df.columns = df.columns.str.lower().str.replace('-', '_')

# changing the appointmentid
df['patientid'] = df['patientid'].astype(int)

# Convert to datetime
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
# Convert to datetime
df['appointmentday'] = pd.to_datetime(df['appointmentday'])

# encoding the target column
df.no_show = (df.no_show == 'Yes').astype(int)

# So we will combine the categories 1,2,3 & 4 as 1 (True for being handicap)
df['handcap'] = np.where(df['handcap'].isin([1, 2, 3]), 1, 0)


df['scheduledday_date'] = df['scheduledday'].dt.day
df['scheduledday_month'] = df['scheduledday'].dt.month
df['scheduledday_dow'] = df['scheduledday'].dt.dayofweek + 1 # Monday is should be identified as 1
df['scheduledday_hr'] = df['scheduledday'].dt.hour
df['appointmentday_date'] = df['appointmentday'].dt.day
df['appointmentday_month'] = df['appointmentday'].dt.month
df['appointmentday_dow'] = df['appointmentday'].dt.dayofweek + 1  # Monday is should be identified as 1


#convert the existing to datetime dtype without the timestamp
# Strip the time component but keep as datetime64[ns]
df['scheduledday'] = pd.to_datetime(df['scheduledday'], errors='coerce').dt.normalize()
df['appointmentday'] = pd.to_datetime(df['appointmentday'], errors='coerce').dt.normalize()

# add the feature for waiting period for appointmentday from scheduledday
df['days_to_wait'] = (df['appointmentday'] - df['scheduledday']).dt.days

# dropping these record that has appointmentday before scheduledday
df = df.drop(df[(df['appointmentday'] - df['scheduledday']).dt.days < 0].index)


# remove the timestamp from the dates
df['scheduledday'] = pd.to_datetime(df['scheduledday']).dt.date
df['appointmentday'] = pd.to_datetime(df['appointmentday']).dt.date

# convert it back from the object dtype to datetime dtype from previous line(s)
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
df['appointmentday'] = pd.to_datetime(df['appointmentday'])

# Ensure the dataset is sorted by patientid and scheduledday
df = df.sort_values(by=['patientid', 'scheduledday'])

# Group by patientid and calculate cumulative counts
df['previous_appointments'] = df.groupby('patientid').cumcount()

# Calculate cumulative missed appointments for each patient, resetting index to align with df
df['missed_appointments'] = df.groupby('patientid')['no_show'].apply(
    lambda x: x.shift().cumsum()
).reset_index(level=0, drop=True) # Reset index to align with df

# Fill NaN values for the first appointment (no previous history)
df['missed_appointments'] = df['missed_appointments'].fillna(0).astype(int)

# Adding sum count of the health issues a patient has who had made an appointment.
df['cum_healthissues'] = df[['hipertension', 'diabetes', 'alcoholism', 'handcap']].sum(axis=1)

# Dropping the age outliers: 115 and -1 years
df = df.drop(df[(df['age'] == 115) | (df['age'] == -1)].index)

# save the current state of the cleaned, prepared and featured engineered
# dataframe df to be used by the test.py for tranformation of raw patient/appointment 
# data from the test dataset to be suitable for input to the model.
df.to_pickle('cleaned_prepared_df.pkl')

# as seen above indeces are not squential and so we need to reset them
df.reset_index(drop=True, inplace=True)

# patientid & appointmentid are identifiers
# all necessary information from 'scheduledday', 'appointmentday' were already extracted and feature engineered
# there are 81 unique values values of neighbourhood and the large categorical range in the feature may cause overfitting in the model
drop_columns = ['patientid', 'appointmentid', 'neighbourhood', 'scheduledday', 'appointmentday']

# as such, these columns are being dropped
df.drop(drop_columns, axis=1, inplace=True)

# I choose F as 1 because number females are almost twice that of males and so more 1s & lesser 0s would be generated.
#Lesser zeros would prevent the increase of sparse-ness in the dataset
df.gender = (df.gender == 'F').astype(int)

#Splitting the dataset into full train (for training with cross validation) and test set : 20% test, 80% full train
# as the target class data is imbalanced [roughly 80% show and 20% no show],
#we use stratify to maintain the same proportion of positive class labels (0-->show & 1-->no_show) in the individual splits to mitigate imbalance effect while trainning the model
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['no_show'])

# setting the target dataset for full_train and tesr
y_full_train = df_full_train.no_show.values
y_test = df_test.no_show.values

# removing the target feature from the full_train and test dataset
del df_full_train['no_show']
del df_test['no_show']

# choosing the best evaluated model to train

# Convert the full train and test dataset to DMatrix
dtrain_final = xgb.DMatrix(df_full_train, label=y_full_train)
dtest_final = xgb.DMatrix(df_test, label=y_test)


# setting the xgb_params with evaluated tuned parameters
xgb_params_final = {
 'eta': 0.1,
 'max_depth': 6,
 'min_child_weight': 1,
 'scale_pos_weight': 3.9531635168447,
 'objective': 'binary:logistic',
 'eval_metric': 'auc',
 'nthread': 8,
 'seed': 42,
 'verbosity': 1
}

# Train the model on the full train dataset
model_final = xgb.train(
    params=xgb_params_final,
    dtrain=dtrain_final,
    num_boost_round=110, # as decided earlier
    evals=[(dtrain_final, 'train_final'), (dtest_final, 'test')], # Use the tuned number of boosting rounds
    verbose_eval=10
)

# saving the model
with open('model_final.bin', 'wb') as file_out:
    pickle.dump(model_final, file_out)
file_out.close()
print('The model has been trained and saved as --> model_final.bin')