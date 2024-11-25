import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import DMatrix, train
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_final.bin'

# Load the model
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
    print('Model deployed..!')
f_in.close()



app = Flask('no_show_predict')


@app.route('/predict', methods=['POST'])
def predict():
    appointment = request.get_json()
    print("Inside the web service ....... ")

    # feature drop to align with model input
    keys_to_drop = ['patientid', 'appointmentid', 'neighbourhood']  # Replace with the actual keys
    for key in keys_to_drop:
        appointment.pop(key, None)  # Use pop to avoid KeyError if the key doesn't exist

    #print("modified now... ", appointment)

    # Convert the appointment dictionary to a DataFrame (1 row, multiple columns)
    df_test = pd.DataFrame([appointment])
    #print("Dataframe created for prediction:\n", df_test)

    # Create a DMatrix from the DataFrame (no labels, since it's a prediction)
    dtest = xgb.DMatrix(df_test)  # Only features, no labels
    

    #print('all good uptil here.....................!!!')
    # Predict using the model (output is the probability of the positive class, class 1)
    y_pred = model.predict(dtest)[0]  # Predict probabilities (assuming binary classification)


    # Determine if it's a no-show based on a threshold (e.g., 0.5)
    is_no_show = (y_pred >= 0.5)
    y_pred_rounded = round(y_pred, 3)

    #print('No-Show Probability:', round(y_pred, 3))
    #print('Would it be a No-Show:', is_no_show)

    result = {
        'no_show_probability' : float(y_pred_rounded),
        'is_no_show' : bool(is_no_show)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
