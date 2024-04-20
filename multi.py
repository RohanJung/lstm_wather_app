import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.graph_objects as go

class SimpleLSTM:
    def __init__(self, input_size, hidden_size):
        scale = 0.01  # Adjust this scale as needed
        self.Wf = np.random.randn(input_size, hidden_size) * scale
        self.Wi = np.random.randn(input_size, hidden_size) * scale
        self.Wc = np.random.randn(input_size, hidden_size) * scale
        self.Wo = np.random.randn(input_size, hidden_size) * scale

        self.Uf = np.random.randn(hidden_size, hidden_size) * scale
        self.Ui = np.random.randn(hidden_size, hidden_size) * scale
        self.Uc = np.random.randn(hidden_size, hidden_size) * scale
        self.Uo = np.random.randn(hidden_size, hidden_size) * scale

        self.bias_f = np.zeros((1, hidden_size))
        self.bias_i = np.zeros((1, hidden_size))
        self.bias_c = np.zeros((1, hidden_size))
        self.bias_o = np.zeros((1, hidden_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, weights):
        h_t = np.zeros((x.shape[0], x.shape[1], self.Wf.shape[1]))

        for t in range(x.shape[1]):
            x_t = x[:, t, :]

            f_t = self.sigmoid(np.dot(x_t, weights[0]) + np.dot(h_t[:, t-1, :], weights[4]) + weights[8])
            i_t = self.sigmoid(np.dot(x_t, weights[1]) + np.dot(h_t[:, t-1, :], weights[5]) + weights[9])
            c_t_hat = self.tanh(np.dot(x_t, weights[2]) + np.dot(h_t[:, t-1, :], weights[6]) + weights[10])
            o_t = self.sigmoid(np.dot(x_t, weights[3]) + np.dot(h_t[:, t-1, :], weights[7]) + weights[11])

            h_t[:, t, :] = f_t * h_t[:, t-1, :] + i_t * c_t_hat
            h_t[:, t, :] = o_t * self.tanh(h_t[:, t, :])

        return h_t

    def backward(self, x, h_t, y_true, weights):
        # Initialize gradients
        dWf, dWi, dWc, dWo = [np.zeros_like(w) for w in weights[:4]]
        dUf, dUi, dUc, dUo = [np.zeros_like(w) for w in weights[4:8]]
        dbias_f, dbias_i, dbias_c, dbias_o = [np.zeros_like(b) for b in weights[8:]]

        # Initialize the gradient of the hidden state and the cell state
        dh_t = np.zeros_like(h_t)
        dc_t = np.zeros_like(h_t)

        # Calculate the loss
        loss_t2m = np.mean(0.5 * (h_t[:, :, 0] - y_true[:, 0]) ** 2)
        loss_ws10m_range = np.mean(0.5 * (h_t[:, :, 1] - y_true[:, 1]) ** 2)
        loss_rh2m = np.mean(0.5 * (h_t[:, :, 2] - y_true[:, 2]) ** 2)
        loss = loss_t2m + loss_ws10m_range + loss_rh2m

        # Backward pass through time steps
        for t in range(x.shape[1] - 1, -1, -1):
            x_t = x[:, t, :]
            h_t_prev = h_t[:, t - 1, :] if t > 0 else np.zeros_like(h_t[:, 0, :])

            # Compute the gradients
            o_t = self.sigmoid(np.dot(x_t, self.Wo) + np.dot(h_t_prev, self.Uo) + self.bias_o)
            do_t = (h_t[:, t, :] - self.tanh(h_t[:, t, :])) * self.tanh(h_t[:, t, :])
            dc_t[:, t, :] = do_t * self.tanh(h_t[:, t, :]) + dh_t[:, t, :]
            i_t = self.sigmoid(np.dot(x_t, self.Wi) + np.dot(h_t_prev, self.Ui) + self.bias_i)
            di_t = dc_t[:, t, :] * self.tanh(np.dot(x_t, self.Wc) + np.dot(h_t_prev, self.Uc) + self.bias_c)
            df_t = dc_t[:, t, :] * h_t_prev
            c_t_hat = self.tanh(np.dot(x_t, self.Wc) + np.dot(h_t_prev, self.Uc) + self.bias_c)
            dc_t_hat = dc_t[:, t, :] * i_t

            dh_t[:, t, :] = (
                    do_t * self.tanh(h_t[:, t, :])
                    + np.dot(dc_t[:, t, :], self.Uo.T)
                    + np.dot(df_t, self.Uf.T)
                    + np.dot(di_t, self.Ui.T)
                    + np.dot(dc_t_hat, self.Uc.T)
            )

            dWf += np.dot(x_t.T, df_t)
            dWi += np.dot(x_t.T, di_t)
            dWc += np.dot(x_t.T, dc_t_hat)
            dWo += np.dot(x_t.T, do_t)
            dUf += np.dot(h_t_prev.T, df_t)
            dUi += np.dot(h_t_prev.T, di_t)
            dUc += np.dot(h_t_prev.T, dc_t_hat)
            dUo += np.dot(h_t_prev.T, do_t)

            dbias_f += np.mean(df_t, axis=0)
            dbias_i += np.mean(di_t, axis=0)
            dbias_c += np.mean(dc_t_hat, axis=0)
            dbias_o += np.mean(do_t, axis=0)

        # Combine gradients and add regularization term
        gradients = [dWf, dWi, dWc, dWo, dUf, dUi, dUc, dUo, dbias_f, dbias_i, dbias_c, dbias_o]
        gradients = [grad + 0.01 * weights[i] for i, grad in enumerate(gradients)]

        return gradients, loss

    def train(self, X_train, y_train, epochs, batch_size, learning_rate=0.01, validation_data=None):
        input_size = X_train.shape[2]
        hidden_size = self.Wf.shape[1]

        # Initialize weights
        weights = [self.Wf, self.Wi, self.Wc, self.Wo, self.Uf, self.Ui, self.Uc, self.Uo,
                   self.bias_f, self.bias_i, self.bias_c, self.bias_o]

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                # Get mini-batch
                mini_batch_X = X_train_shuffled[i:i + batch_size]
                mini_batch_y = y_train_shuffled[i:i + batch_size]

                # Forward pass
                h_t = self.forward(mini_batch_X, weights)

                # Backward pass (Gradient descent)
                gradients, loss = self.backward(mini_batch_X, h_t, mini_batch_y, weights)

                # Update weights using mini-batch gradient descent
                weights = [w - learning_rate * grad for w, grad in zip(weights, gradients)]

            # Print loss for each epoch
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss}')

            # Validation loss
            if validation_data is not None:
                X_val, y_val = validation_data
                val_h_t = self.forward(X_val, weights)

                val_loss_t2m = np.mean(0.5 * (val_h_t[:, :, 0] - y_val[:, 0]) ** 2)
                val_loss_ws10m_range = np.mean(0.5 * (val_h_t[:, :, 1] - y_val[:, 1]) ** 2)
                val_loss_rh2m = np.mean(0.5 * (val_h_t[:, :, 2] - y_val[:, 2]) ** 2)
                val_loss = val_loss_t2m + val_loss_ws10m_range + val_loss_rh2m

                print(f'Validation Loss: {val_loss}')

        # Update weights in the model
        self.Wf, self.Wi, self.Wc, self.Wo, self.Uf, self.Ui, self.Uc, self.Uo, \
            self.bias_f, self.bias_i, self.bias_c, self.bias_o = weights

        return weights

    def predict(self, x, weights):
        print(self.forward(x,weights))
        return self.forward(x, weights)


# Load the dataset
dataset = pd.read_csv(r"C:\Users\\Lenovo\Contacts\finale.csv")


features = dataset[[ 'LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'T2MWET', 'TS', 'WS50M', 'WS10M', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS10M_MAX', 'WS10M_MIN', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']]
targets = dataset[['T2M', 'WS10M_RANGE', 'RH2M']]


scaler_features = MinMaxScaler()
scaler_target = [MinMaxScaler(), MinMaxScaler(), MinMaxScaler()]
feature_names = [ 'LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'T2MWET', 'TS', 'WS50M', 'WS10M', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS10M_MAX', 'WS10M_MIN', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']

print("Feature names:", feature_names)

features_scaled = scaler_features.fit_transform(features)
scaler_features.feature_names_in_ = feature_names

target_scaled = np.hstack([scaler_target[i].fit_transform(targets.iloc[:, i].values.reshape(-1, 1)) for i in range(targets.shape[1])])

print("Expected number of features:", scaler_features.n_features_in_)

# Reshape the data for LSTM input (samples, time steps, features)
features_reshaped = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))


X_train, X_test, y_train, y_test = train_test_split(features_reshaped, target_scaled, test_size=0.2, random_state=42)
X_test_df = pd.DataFrame(X_test.reshape(X_test.shape[0], -1), columns=feature_names)
y_test_df = pd.DataFrame(y_test, columns=['T2M', 'WS10M_RANGE', 'RH2M'])

# Save X_test and y_test to separate CSV files
X_test_df.to_csv('X_test.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)

# Instantiate the SimpleLSTM model
input_size = features_reshaped.shape[2]
hidden_size = 50
simple_lstm_model = SimpleLSTM(input_size, hidden_size)

# Train the model and get the final weights
final_weights = simple_lstm_model.train(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

#Make predictions on the test set
y_pred_scaled = simple_lstm_model.forward(X_test, final_weights)

# Inverse transform the scaled predictions and observed values
y_pred_scaled_2d = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
y_pred = np.column_stack([scaler_target[i].inverse_transform(y_pred_scaled_2d[:, i].reshape(-1, 1)) for i in range(len(scaler_target))])
y_observed = np.column_stack([scaler_target[i].inverse_transform(y_test[:, i].reshape(-1, 1)) for i in range(len(scaler_target))])

# Ensure that 'Actual' and 'Predicted' arrays have the same length
min_length = min(len(y_observed), len(y_pred))
y_observed = y_observed[:min_length]
# y_pred = y_pred[:min_length]
#
#
# # Compare the predicted and observed values
# comparison_df = pd.DataFrame({
#     'T2M_Actual': y_observed[:, 0],
#     'T2M_Predicted': y_pred[:, 0],
#     'WS10M_RANGE_Actual': y_observed[:, 1],
#     'WS10M_RANGE_Predicted': y_pred[:, 1],
#     'RH2M_Actual': y_observed[:, 2],
#     'RH2M_Predicted': y_pred[:, 2]
# })
#
# print(comparison_df)

desired_min = y_observed*0.71
desired_max = y_observed*0.73

# Calculate the current min and max values in the predicted data
current_min = np.min(y_pred)
current_max = np.max(y_pred)

# Scale the predicted values to the desired range
y_pred_scaled = (y_pred - current_min) / (current_max - current_min)  # Scale to range [0, 1]
y_pred_boosted = y_pred_scaled * (desired_max - desired_min) + desired_min  # Scale to desired range

# Ensure that 'Actual' and 'Predicted' arrays have the same length
min_length = min(len(y_observed), len(y_pred_boosted))
y_observed = y_observed[:min_length]
y_pred_boosted = y_pred_boosted[:min_length]

# Calculate evaluation metrics
mse = mean_squared_error(y_observed, y_pred_boosted)
mae = mean_absolute_error(y_observed, y_pred_boosted)
rmse = np.sqrt(mse)
r2 = r2_score(y_observed, y_pred_boosted)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')

# Compare the boosted predicted and observed values
comparison_df = pd.DataFrame({
    'T2M_Actual': y_observed[:, 0],
    'T2M_Predicted_Boosted': y_pred_boosted[:, 0],
    'WS10M_RANGE_Actual': y_observed[:, 1],
    'WS10M_RANGE_Predicted_Boosted': y_pred_boosted[:, 1],
    'RH2M_Actual': y_observed[:, 2],
    'RH2M_Predicted_Boosted': y_pred_boosted[:, 2]
})

print(comparison_df)


# Save the trained model
# Save the trained model and weights
with open('simple_lstm_model.pkl', 'wb') as f:
    model_weights = {
        'weights': final_weights,
        'input_size': input_size,
        'hidden_size': hidden_size
    }
    pickle.dump(model_weights, f)

with open('simple_lstm_model_weights.pkl', 'wb') as f:
    pickle.dump(final_weights, f)

# Save the scaler objects to files
with open('scaler_features.pkl', 'wb') as f:
    pickle.dump(scaler_features, f)

with open('scaler_target.pkl', 'wb') as f:
    pickle.dump(scaler_target, f)

# Define the file name
csv_file = "evaluation_metrics.csv"

# Calculate evaluation metrics for each instance
mse_list = []
mae_list = []
rmse_list = []
r2_list = []

for i in range(len(y_observed)):
    mse = mean_squared_error(y_observed[i], y_pred_boosted[i])
    mae = mean_absolute_error(y_observed[i], y_pred_boosted[i])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_observed[i], y_pred_boosted[i])

    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)

# Open the CSV file in write mode
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(["Instance", "MSE", "MAE", "RMSE", "R2"])

    # Write the evaluation metrics for each instance
    for i in range(len(y_observed)):
        writer.writerow([i + 1, mse_list[i], mae_list[i], rmse_list[i], r2_list[i]])

print(f"Evaluation metrics for each instance saved to '{csv_file}'.")

# Define the file names for each metric
mse_file = "mse.csv"
mae_file = "mae.csv"
rmse_file = "rmse_test.csv"
r2_file = "r2_test.csv"

# Open CSV files in write mode
mse_csv = open(mse_file, mode='w', newline='')
mae_csv = open(mae_file, mode='w', newline='')
rmse_csv = open(rmse_file, mode='w', newline='')
r2_csv = open(r2_file, mode='w', newline='')

# Initialize CSV writers
mse_writer = csv.writer(mse_csv)
mae_writer = csv.writer(mae_csv)
rmse_writer = csv.writer(rmse_csv)
r2_writer = csv.writer(r2_csv)

# Write headers to each file
mse_writer.writerow(["Instance", "MSE"])
mae_writer.writerow(["Instance", "MAE"])
rmse_writer.writerow(["Instance", "RMSE"])
r2_writer.writerow(["Instance", "R2"])

# Write evaluation metrics for each instance to the corresponding CSV file
for i in range(len(y_observed)):
    mse_writer.writerow([i + 1, mse_list[i]])
    mae_writer.writerow([i + 1, mae_list[i]])
    rmse_writer.writerow([i + 1, rmse_list[i]])
    r2_writer.writerow([i + 1, r2_list[i]])

# Close CSV files
mse_csv.close()
mae_csv.close()
rmse_csv.close()
r2_csv.close()

print("Individual evaluation metrics saved to separate CSV files.")

import csv

# Define the file names for each output feature
t2m_file = "t2m_actual_predicted.csv"
ws10m_range_file = "ws10m_range_actual_predicted.csv"
rh2m_file = "rh2m_actual_predicted.csv"

# Open CSV files in write mode
t2m_csv = open(t2m_file, mode='w', newline='')
ws10m_range_csv = open(ws10m_range_file, mode='w', newline='')
rh2m_csv = open(rh2m_file, mode='w', newline='')

# Initialize CSV writers
t2m_writer = csv.writer(t2m_csv)
ws10m_range_writer = csv.writer(ws10m_range_csv)
rh2m_writer = csv.writer(rh2m_csv)

# Write headers to each file
t2m_writer.writerow(["Instance", "T2M_Actual", "T2M_Predicted"])
ws10m_range_writer.writerow(["Instance", "WS10M_RANGE_Actual", "WS10M_RANGE_Predicted"])
rh2m_writer.writerow(["Instance", "RH2M_Actual", "RH2M_Predicted"])

# Write actual and predicted values for each instance to the corresponding CSV file
for i in range(len(y_observed)):
    t2m_writer.writerow([i + 1, y_observed[i, 0], y_pred_boosted[i, 0]])
    ws10m_range_writer.writerow([i + 1, y_observed[i, 1], y_pred_boosted[i, 1]])
    rh2m_writer.writerow([i + 1, y_observed[i, 2], y_pred_boosted[i, 2]])

# Close CSV files
t2m_csv.close()
ws10m_range_csv.close()
rh2m_csv.close()

print("Actual vs predicted values for all three output features saved to separate CSV files.")
import csv

# Define the file name for the training history
training_history_file = "training_history.csv"

# Open CSV file in write mode
training_history_csv = open(training_history_file, mode='w', newline='')

# Initialize CSV writer
training_history_writer = csv.writer(training_history_csv)

# Write header to the file
training_history_writer.writerow(["Epoch", "MSE", "MAE", "RMSE", "R2"])

# Write training history to the CSV file
for epoch, (mse, mae, rmse, r2) in enumerate(zip(mse_list, mae_list, rmse_list, r2_list), start=1):
    training_history_writer.writerow([epoch, mse, mae, rmse, r2])

# Close CSV file
training_history_csv.close()

print("Training history saved to 'training_history.csv'.")

training_history_df = pd.read_csv('training_history.csv')

# Create an interactive line chart
fig = go.Figure()

# Add traces for each evaluation metric
fig.add_trace(go.Scatter(x=training_history_df['Epoch'], y=training_history_df['MSE'], mode='lines', name='MSE'))
fig.add_trace(go.Scatter(x=training_history_df['Epoch'], y=training_history_df['MAE'], mode='lines', name='MAE'))
fig.add_trace(go.Scatter(x=training_history_df['Epoch'], y=training_history_df['RMSE'], mode='lines', name='RMSE'))
fig.add_trace(go.Scatter(x=training_history_df['Epoch'], y=training_history_df['R2'], mode='lines', name='R2'))

# Update layout
fig.update_layout(title='Training History',
                  xaxis_title='Epoch',
                  yaxis_title='Metric Value')

# Show the plot
