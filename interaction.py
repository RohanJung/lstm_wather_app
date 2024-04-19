import plotly.graph_objects as go
import pandas as pd

# Create training history plot
# Load data
comparison_result_df = pd.read_csv("comparison_results.csv")
evaluation_metrics_df = pd.read_csv("evaluation_metrics.csv")
mae_df = pd.read_csv("mae.csv")
mse_df = pd.read_csv("mse.csv")
r2_test_df = pd.read_csv("r2_test.csv")
rmse_df = pd.read_csv("rmse_test.csv")
training_history_df = pd.read_csv("training_history.csv")
rh2m_actual_predicted_df = pd.read_csv("rh2m_actual_predicted.csv")
t2m_acutal_predicted_df = pd.read_csv("t2m_actual_predicted.csv")
ws10m_range_actual_predicted_df = pd.read_csv("ws10m_range_actual_predicted.csv")



training_history_fig = go.Figure()
training_history_fig.add_trace(go.Scatter(x=training_history_df['Epoch'], y=training_history_df['MSE'], mode='lines', name='MSE'))
training_history_fig.add_trace(go.Scatter(x=training_history_df['Epoch'], y=training_history_df['MAE'], mode='lines', name='MAE'))
training_history_fig.add_trace(go.Scatter(x=training_history_df['Epoch'], y=training_history_df['RMSE'], mode='lines', name='RMSE'))
training_history_fig.add_trace(go.Scatter(x=training_history_df['Epoch'], y=training_history_df['R2'], mode='lines', name='R2'))
training_history_fig.update_layout(title='Training History Metrics', xaxis_title='Epoch', yaxis_title='Metrics')

# Save training history plot as HTML
training_history_fig.write_html("training_history_graph.html")

# Create comparison result plot
comparison_result_fig = go.Figure()
comparison_result_fig.add_trace(go.Scatter(x=comparison_result_df.index, y=comparison_result_df['T2M_Actual'], mode='lines', name='T2M Actual'))
comparison_result_fig.add_trace(go.Scatter(x=comparison_result_df.index, y=comparison_result_df['T2M_Predicted'], mode='lines', name='T2M Predicted'))
comparison_result_fig.add_trace(go.Scatter(x=comparison_result_df.index, y=comparison_result_df['WS10M_RANGE_Actual'], mode='lines', name='WS10M_RANGE Actual'))
comparison_result_fig.add_trace(go.Scatter(x=comparison_result_df.index, y=comparison_result_df['WS10M_RANGE_Predicted'], mode='lines', name='WS10M_RANGE Predicted'))
comparison_result_fig.add_trace(go.Scatter(x=comparison_result_df.index, y=comparison_result_df['RH2M_Actual'], mode='lines', name='RH2M Actual'))
comparison_result_fig.add_trace(go.Scatter(x=comparison_result_df.index, y=comparison_result_df['RH2M_Predicted'], mode='lines', name='RH2M Predicted'))
comparison_result_fig.update_layout(title='Comparison Result', xaxis_title='Instance', yaxis_title='Values')

# Save comparison result plot as HTML
comparison_result_fig.write_html("comparison_result_graph.html")

# Create evaluation metrics plot
evaluation_metrics_fig = go.Figure()
evaluation_metrics_fig.add_trace(go.Bar(x=evaluation_metrics_df['Instance'], y=evaluation_metrics_df['MSE'], name='MSE'))
evaluation_metrics_fig.add_trace(go.Bar(x=evaluation_metrics_df['Instance'], y=evaluation_metrics_df['MAE'], name='MAE'))
evaluation_metrics_fig.add_trace(go.Bar(x=evaluation_metrics_df['Instance'], y=evaluation_metrics_df['RMSE'], name='RMSE'))
evaluation_metrics_fig.add_trace(go.Bar(x=evaluation_metrics_df['Instance'], y=evaluation_metrics_df['R2'], name='R2'))
evaluation_metrics_fig.update_layout(title='Evaluation Metrics', xaxis_title='Instance', yaxis_title='Metrics')

# Save evaluation metrics plot as HTML
evaluation_metrics_fig.write_html("evaluation_metrics_graph.html")

print("HTML files for training history, comparison result, and evaluation metrics have been created.")
