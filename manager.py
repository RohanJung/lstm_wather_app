import pandas as pd
import plotly.express as px

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

# Create interactive graphs for evaluation metrics
mse_fig = px.line(mse_df, x="Instance", y="MSE", title="Mean Squared Error (MSE) over Instances")
mae_fig = px.line(mae_df, x="Instance", y="MAE", title="Mean Absolute Error (MAE) over Instances")
rmse_fig = px.line(rmse_df, x="Instance", y="RMSE", title="Root Mean Squared Error (RMSE) over Instances")
r2_test_fig = px.line(r2_test_df, x="Instance", y="R2", title="R-squared (R2) over Instances")


# Create interactive graphs for comparison between actual and predicted values
t2m_fig = px.line(t2m_acutal_predicted_df, x="Instance", y=["T2M_Actual", "T2M_Predicted"], title="Comparison of T2M Actual vs Predicted")
rh2m_fig = px.line(rh2m_actual_predicted_df, x="Instance", y=["RH2M_Actual", "RH2M_Predicted"], title="Comparison of RH2M Actual vs Predicted")
ws10m_range_fig = px.line(ws10m_range_actual_predicted_df, x="Instance", y=["WS10M_RANGE_Actual", "WS10M_RANGE_Predicted"], title="Comparison of WS10M_RANGE Actual vs Predicted")

# Save interactive graphs as HTML files
mse_fig.write_html("mse_graph.html")
mae_fig.write_html("mae_graph.html")
rmse_fig.write_html("rmse_graph.html")
r2_test_fig.write_html("r2_graph.html")
t2m_fig.write_html("t2m_comparison_graph.html")
rh2m_fig.write_html("rh2m_comparison_graph.html")
ws10m_range_fig.write_html("ws10m_range_comparison_graph.html")


print("Interactive graphs saved as HTML files.")
