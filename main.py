import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

# Load the Brooklyn homes dataset
brooklyn_homes = pd.read_csv("brooklyn_sales_map.csv")
inflation_data = pd.read_csv("Inflation.csv")
percapita_data = pd.read_csv("Percapita.csv")

# Exclude transactions with sale_price of $0 or nominal amounts
brooklyn_homes = brooklyn_homes[brooklyn_homes.sale_price > 1000]

# Group the data by year and calculate the total sales for each year
sales_by_year = brooklyn_homes.groupby(brooklyn_homes['sale_date'].str[:4])['sale_price'].sum()

# Plot the data
plt.plot(sales_by_year.index, sales_by_year.values)
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.title('Comparison of Sales Over the Years')
plt.show()

# Extract year and month from the sale_date column
brooklyn_homes['sale_date'] = pd.to_datetime(brooklyn_homes['sale_date'])
brooklyn_homes['Year'] = brooklyn_homes['sale_date'].dt.year
brooklyn_homes['Month'] = brooklyn_homes['sale_date'].dt.month

# Group the data by year and month, and calculate the total sales for each month
sales_by_month = brooklyn_homes.groupby(['Year', 'Month'])['sale_price'].sum()

# Plot the data with different colored lines for each year
fig, ax = plt.subplots(figsize=(10, 6))

for year in brooklyn_homes['Year'].unique():
    sales_data_year = sales_by_month.loc[year]
    ax.plot(sales_data_year.index.get_level_values('Month'), sales_data_year.values, label=f'Year {year}')

ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_xlabel('Month')
ax.set_ylabel('Total Sales')
ax.set_title('Monthly Sales Comparison Over the Years')
ax.legend()
plt.show()

sales_by_neighborhood = brooklyn_homes.groupby('neighborhood')['sale_price'].sum()

# Sort neighborhoods by total sales in descending order
sales_by_neighborhood = sales_by_neighborhood.sort_values(ascending=False)

# Plot sales per neighborhood
plt.figure(figsize=(12, 8))
sales_by_neighborhood.plot(kind='bar', color='skyblue')
plt.xlabel('Neighborhood')
plt.ylabel('Total Sales')
plt.title('Total Sales per Neighborhood in Brooklyn')
plt.xticks(rotation=90)
plt.show()

# Group the data by year and calculate the average sale_price for each year
avg_price_by_year = brooklyn_homes.groupby('Year')['sale_price'].mean()

# Plot the comparison of sale_price variation over the years
plt.figure(figsize=(10, 6))
plt.plot(avg_price_by_year.index, avg_price_by_year.values, marker='o', color='orange', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('Year')
plt.ylabel('Average Sale Price')
plt.title('Average Sale Price Variation Over the Years')
plt.grid(True)
plt.show()


# Sort the data by sale_price in descending order and select top 100 highest prices
top_100_highest_prices = brooklyn_homes.nlargest(100, 'sale_price')

# Group the top 100 highest prices by neighborhood and calculate the average price for each neighborhood
avg_price_top_100_by_neighborhood = top_100_highest_prices.groupby('neighborhood')['sale_price'].mean()

# Select the top 10 neighborhoods with the highest average prices from the top 100 highest prices
top_10_neighborhoods = avg_price_top_100_by_neighborhood.nlargest(10)

# Plot the top 10 neighborhoods and their average prices
plt.figure(figsize=(12, 6))
top_10_neighborhoods.plot(kind='bar', color='skyblue')
plt.xlabel('Neighborhood')
plt.ylabel('Average Sale Price')
plt.title('Top 10 Neighborhoods with Highest Average Sale Prices (Top 100 Highest Sale Price Houses)')
plt.xticks(rotation=45)
plt.show()

# Group the data by year and find the highest and lowest sale prices for each year
highest_prices = brooklyn_homes.groupby('Year')['sale_price'].max()
lowest_prices = brooklyn_homes.groupby('Year')['sale_price'].min()

# Plot the highest and lowest sale prices over the years
plt.figure(figsize=(10, 6))
plt.plot(highest_prices.index, highest_prices.values, marker='o', color='orange', label='Highest Price')
plt.plot(lowest_prices.index, lowest_prices.values, marker='o', color='green', label='Lowest Price (> $1000)')
plt.xlabel('Year')
plt.ylabel('Sale Price')
plt.title('Highest and Lowest Sale Prices Over the Years')
plt.legend()
plt.grid(True)
plt.show()

# Group the data by year and calculate the total number of homes sold per year
homes_sold_per_year = brooklyn_homes.groupby('Year').size().reset_index(name='Homes Sold')

# Ensure that 'Year' column is in both DataFrames and has the same name
inflation_data['Year'] = inflation_data['Year'].astype(int)

# Merge inflation data with homes sold data on 'Year' column
merged_data = pd.merge(homes_sold_per_year, inflation_data, on='Year', how='inner')

# Plot the inflation line and the total number of homes sold per year on the same graph
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting total number of homes sold per year (left y-axis)
ax1.plot(merged_data['Year'], merged_data['Homes Sold'], color='orange', marker='o', label='Homes Sold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Homes Sold', color='orange')
ax1.tick_params(axis='y', labelcolor='orange')
ax1.set_title('Inflation and Homes Sold Over the Years')

# Create a second y-axis for inflation (right y-axis)
ax2 = ax1.twinx()
ax2.plot(merged_data['Year'], merged_data['Annual'], color='green', marker='o', label='Inflation')
ax2.set_ylabel('Inflation (Annual %)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Display legends for both lines
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Load and preprocess the time series data
# (Assuming the data is loaded into a DataFrame named 'time_series_data')

# Extract the date column from the dataset (assuming the date column is named 'sale_date')
dates = pd.to_datetime(brooklyn_homes['sale_date'])

# Generate synthetic time series data based on historical data
num_data_points = len(dates)
mean_sale_price = brooklyn_homes['sale_price'].mean()
std_dev_sale_price = brooklyn_homes['sale_price'].std()

# Generate random values following a normal distribution with mean and standard deviation from the actual data
synthetic_sale_prices = np.random.normal(loc=mean_sale_price, scale=std_dev_sale_price, size=num_data_points)

# Create a DataFrame with dates as index and synthetic sale prices
synthetic_time_series_data = pd.DataFrame(data={'sale_price': synthetic_sale_prices}, index=dates)

# Print the generated synthetic time series data
print(synthetic_time_series_data.head())

# Split the data into training and testing sets
train_data, test_data = train_test_split(synthetic_time_series_data, test_size=0.2, random_state=42)

# Train a predictive model (e.g., Random Forest) on the training data
features = train_data.drop(columns=['target_column'])
target = train_data['target_column']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(features, target)

# Make predictions on the test data
predictions = model.predict(test_data.drop(columns=['target_column']))

# Calculate mean squared error to evaluate model performance
mse = mean_squared_error(test_data['target_column'], predictions)
print(f"Mean Squared Error: {mse}")

# Justification for Probabilistic Data Management:
# In time series forecasting, uncertainty is a crucial factor. Probabilistic approaches help us
# quantify this uncertainty. By using probabilistic data management, we can:

# 1. **Handle Noisy Data:** Time series data often contains noise. Probabilistic methods can model
#    and handle this noise, providing a more robust analysis.

# 2. **Estimate Prediction Intervals:** Instead of providing point estimates, probabilistic models
#    allow us to estimate prediction intervals. These intervals convey the range of possible values
#    for the forecasted variable, providing a more realistic understanding of the predictions.

# 3. **Incorporate Prior Knowledge:** Probabilistic approaches enable the incorporation of prior
#    knowledge and domain expertise into the forecasting process, leading to more accurate results.

# 4. **Detect Anomalies:** Probabilistic models can identify anomalies and outliers in the time
#    series data, helping in detecting unusual patterns and improving the overall data quality.

# Example: Calculate 95% prediction interval for test data using Normal distribution assumption
mean_predictions = np.mean(predictions)
std_dev_predictions = np.std(predictions)
confidence_level = 0.95
z_score = norm.ppf((1 + confidence_level) / 2)
lower_bound = mean_predictions - z_score * std_dev_predictions
upper_bound = mean_predictions + z_score * std_dev_predictions

print(f"95% Prediction Interval: [{lower_bound}, {upper_bound}]")

# Further enhance the model using probabilistic techniques like Bayesian methods or quantile regression
# to better capture uncertainty and improve the accuracy of the time series forecasting.