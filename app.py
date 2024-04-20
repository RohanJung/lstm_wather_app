import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

# Import the SimpleLSTM class from lstm module
from multi import SimpleLSTM, input_size, hidden_size, feature_names

app = Flask(__name__)

# Load the scaler objects
with open('scaler_features.pkl', 'rb') as f:
    scaler_features = pickle.load(f)

with open('scaler_target.pkl', 'rb') as f:
    scaler_target = pickle.load(f)

# Function to preprocess input data
# Function to preprocess input data
print("Feature names during fitting:", feature_names)
def preprocess_input(data):
    try:
        # Check if the number of features matches the expected number
        if data.shape[1] != scaler_features.n_features_in_:
            raise ValueError(f"Input data has {data.shape[1]} features, but MinMaxScaler is expecting {scaler_features.n_features_in_} features.")

        # Scale the input features
        print('Scaling input features...')
        scaled_input = scaler_features.transform(data)
        print('Scaled Input Data:', scaled_input)

        # Reshape the data for LSTM input
        reshaped_input = np.reshape(scaled_input, (scaled_input.shape[0], 1, scaled_input.shape[1]))
        print('Reshaped Input Data:', reshaped_input)

        return reshaped_input
    except Exception as e:
        print("Preprocessing Input Error:", str(e))
        raise e

# Route for index page


@app.route('/')
def index():
    return render_template('main.html', feature_names=feature_names)

@app.route('/dash')
def dashbaord():
    return render_template('dash.html')
@app.route('/comparison_result_graph')
def comparison_result_graph():
    return render_template('comparison_result_graph.html')

@app.route('/evaluation_metrics_graph')
def evaluation_metrics_graph():
    return render_template('evaluation_metrics_graph.html')

@app.route('/mse_graph')
def mse_graph():
    return render_template('mse_graph.html')

@app.route('/rmse_graph')
def rmse_graph():
    return render_template('rmse_graph.html')

@app.route('/t2m_comparison_graph')
def rmse_graph():
    return render_template('t2m_comparison_graph.html')

@app.route('/rh2m_comparison_graph')
def rh2m_comparison_graph():
    return render_template('rh2m_comparison_graph.html')

@app.route('/ws10m_range_comparison_graph')
def ws10m_range_comparison_graph():
    return render_template('ws10m_range_comparison_graph.html')


@app.route('/work')
def work():
    return  render_template('work.html')

@app.route('/test_dataset')
def test_dataset():
    return  render_template('test_dataset.html')

@app.route('/gg')
def gg():
    return render_template('gg.html')

@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/new')
def main():
    return render_template('new.html')



# Route for prediction
# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the saved model weights
        with open('simple_lstm_model_weights.pkl', 'rb') as f:
            loaded_weights = pickle.load(f)

        # Extract data from the request JSON
        data = request.json
        print("Received Data:", data)

        # Extract the values from the JSON object
        input_values = [float(value) for value in data.values()]
        print("Input Values:", input_values)

        # Convert the input values to a numpy array
        input_data = np.array([input_values])
        print("Input Data:", input_data)

        # Set feature names for the scaler
        scaler_features.feature_names_in_ = feature_names
        print(feature_names)

        # Preprocess input data
        input_data_processed = preprocess_input(input_data)
        print("Preprocessed Input Data:", input_data_processed)
        app.logger.info("Preprocessed Input Data: %s", input_data_processed)

        # Instantiate the SimpleLSTM model
        simple_lstm_model = SimpleLSTM(input_size, hidden_size)

        # Make predictions using the model
        prediction_scaled = simple_lstm_model.forward(input_data_processed, loaded_weights)
        print(prediction_scaled,'this is prediction')

        y_pred_scaled_2d = prediction_scaled.reshape(-1, prediction_scaled.shape[-1])
        y_pred = np.column_stack([scaler_target[i].inverse_transform(y_pred_scaled_2d[:, i].reshape(-1, 1)) for i in
                                  range(len(scaler_target))])
        print(y_pred)
        print('temprature:',y_pred[0][0])
        print('humidity',y_pred[0][1])
        print('windspeed',y_pred[0][2])
        return jsonify({
            'temperature': abs(y_pred[0][0]),'humidity': abs(y_pred[0][1]),'windspeed': abs(y_pred[0][2])
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/new_predict', methods=['POST'])
def new_predict():
    try:
        # Load the saved model weights
        with open('simple_lstm_model_weights.pkl', 'rb') as f:
            loaded_weights = pickle.load(f)

        # Extract latitude and longitude from the request
        lat = float(request.form.get('lat'))
        lon = float(request.form.get('lon'))

        # Generate other input features with a value of 20
        other_features = [20] * 15  # 20 other features with value 20

        # Combine latitude, longitude, and other features
        input_features = [lat, lon] + other_features

        # Convert input features to a numpy array
        input_data = np.array([input_features])

        # Preprocess input data
        input_data_processed = preprocess_input(input_data)

        # Instantiate the SimpleLSTM model
        simple_lstm_model = SimpleLSTM(input_size, hidden_size)

        # Make predictions using the model
        prediction_scaled = simple_lstm_model.forward(input_data_processed, loaded_weights)

        # Inverse transform the predictions
        y_pred_scaled_2d = prediction_scaled.reshape(-1, prediction_scaled.shape[-1])
        y_pred = np.column_stack([scaler_target[i].inverse_transform(y_pred_scaled_2d[:, i].reshape(-1, 1)) for i in range(len(scaler_target))])

        # Extracting output features from the prediction
        temperature = y_pred[0][0]
        humidity = y_pred[0][1]
        windspeed = y_pred[0][2]

        # Return the prediction as a JSON response with 3 output features
        return jsonify({
            'temperature': abs(y_pred[0][0]),'humidity': abs(y_pred[0][1]),'windspeed': abs(y_pred[0][2])
        })

    except Exception as e:
        return jsonify({'error': str(e)})





if __name__ == '__main__':
    app.run(debug=True)
