import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
data = pd.read_csv('data/material_forecasting_data.csv')

# Feature selection
X = data[['project_phase', 'historical_usage', 'material_cost', 'project_duration']]
y = data['predicted_material']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Save model
import joblib
joblib.dump(model, 'material_forecasting_model.pkl')

import pandas as pd
import time

def integrate_real_time_data(sensor_file, output_file):
    while True:
        # Read real-time sensor data
        sensor_data = pd.read_csv(sensor_file)

        # Process and save updated data
        sensor_data['Operational_Status'] = sensor_data['Sensor_Value'].apply(
            lambda x: 'Optimal' if x > 50 else 'Critical'
        )
        sensor_data.to_csv(output_file, index=False)
        print("Real-time data updated...")
        
        # Update every 5 seconds (simulate real-time)
        time.sleep(5)

# Usage
integrate_real_time_data('data/sample_sensor_data.csv', 'data/processed_sensor_data.csv')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('data/material_forecasting_data.csv')

# Visualize KPIs
sns.barplot(x='project_phase', y='predicted_material', data=data)
plt.title('Predicted Material Usage by Project Phase')
plt.savefig('visualizations/kpi_charts.png')
plt.show()

# Cost overrun trends
sns.lineplot(x='project_duration', y='cost_overrun', data=data)
plt.title('Cost Overruns Over Time')
plt.savefig('visualizations/cost_overruns_trend.png')
plt.show()
