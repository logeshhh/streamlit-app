import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Streamlit UI
st.set_page_config(page_title="Facility Asset Predictive Analysis", layout="wide")
st.title("Facility Asset Predictive Analysis Dashboard")
st.markdown("Upload your data to analyze failure patterns and predict the next failure hours for each asset description.")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    
    # Data Overview
    st.markdown("### Dataset Overview")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")
    st.write("Columns:", data.columns.tolist())

    # Ensure data contains 'Asset_Description' and 'Failure_Hours'
    if 'Asset_Description' in data.columns and 'Failure_Hours' in data.columns:
        # Sort data by Asset_Description and Failure_Hours
        data = data.sort_values(by=["Asset_Description", "Failure_Hours"])
        
        # Group by Asset_Description
        grouped_data = data.groupby("Asset_Description")

        # Create a DataFrame to store results
        predictions = []

        for asset_desc, group in grouped_data:
            failure_hours = group["Failure_Hours"].tolist()

            if len(failure_hours) > 1:
                # Create feature and target data for training
                features = []
                target = []
                
                for i in range(1, len(failure_hours)):
                    features.append([failure_hours[i - 1]])  # Previous failure hour
                    target.append(failure_hours[i])  # Current failure hour (target)
                
                # Convert features and target to DataFrame
                X = pd.DataFrame(features, columns=["Previous_Failure_Hour"])
                y = pd.Series(target)
                
                # Split into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Initialize and train the Random Forest Regressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Predict the next failure hour
                predicted_next_failure = model.predict([[failure_hours[-1]]])[0]

                # Calculate mean absolute error for evaluation
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)

                # Append results
                predictions.append({
                    "Asset_Description": asset_desc,
                    "Last_Failure_Hour": failure_hours[-1],
                    "Predicted_Next_Failure_Hour": predicted_next_failure,
                    "MAE": mae
                })

        # Convert the results into a DataFrame
        predictions_df = pd.DataFrame(predictions)

        # Display the predictions
        st.markdown("### Predictions")
        st.write(predictions_df)

        # Combined Graph for Last Failure vs Predicted Next Failure
        st.markdown("### Last Failure Hour vs Predicted Next Failure Hour")

        fig = go.Figure()

        # Add Last Failure Hour with data labels
        fig.add_trace(go.Scatter(
            x=predictions_df["Asset_Description"],
            y=predictions_df["Last_Failure_Hour"],
            mode="lines+markers+text",  # "lines+markers+text" to show both the lines, markers, and text
            name="Last Failure Hour",
            line=dict(color="red"),
            marker=dict(size=8),  # Adjust marker size if needed
            text=predictions_df["Last_Failure_Hour"],  # Display the value of Last_Failure_Hour as text
            textposition="top center"  # Position the text on top of the data points
        ))

        # Add Predicted Next Failure Hour with data labels
        fig.add_trace(go.Scatter(
            x=predictions_df["Asset_Description"],
            y=predictions_df["Predicted_Next_Failure_Hour"],
            mode="lines+markers+text",  # "lines+markers+text" to show both the lines, markers, and text
            name="Predicted Next Failure Hour",
            line=dict(color="blue"),
            marker=dict(size=8),  # Adjust marker size if needed
            text=predictions_df["Predicted_Next_Failure_Hour"],  # Display the value of Predicted_Next_Failure_Hour as text
            textposition="top center"  # Position the text on top of the data points
        ))

        # Update layout
        fig.update_layout(
            title="Comparison of Last Failure Hour vs Predicted Next Failure Hour",
            xaxis_title="Asset Description",
            yaxis_title="Hours",
            template="plotly_dark",
            legend_title="Metrics",
            height=600
        )

        # Display graph
        st.plotly_chart(fig, use_container_width=True)

        st.success("Analysis Complete!")
    else:
        st.error("The dataset must contain 'Asset_Description' and 'Failure_Hours' columns.")
else:
    st.info("Awaiting file upload. Please upload a CSV file.")
