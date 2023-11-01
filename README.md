# Brooklyn-Home-Sales-Analysis

https://github.com/AnusriReddy19/Brooklyn-Home-Sales-Analysis

https://www.kaggle.com/datasets/tianhwu/brooklynhomes2003to2017

Project Overview:
The project aims to analyze and forecast Brooklyn home sales data from 2003 to 2017. The primary dataset is sourced from NYC Housing Sales Data, which was cleaned, merged, and augmented with NYCPluto shapefiles. The goal is to build a time series forecasting model to predict future home sales based on historical data while excluding non-market transactions.

Project Steps:
Data Collection and Preprocessing:

Obtain the NYC Housing Sales Data from the provided link.
Merge and clean the data using VBA scripting in Excel and R.
Augment the dataset with NYCPluto shapefile data using spatial join (left join by "Block" & "Lot").
Handle missing values (NAs) in the augmented dataset appropriately.
Exclude transactions with sale_price of $0 or nominal amounts for market analysis.
Exploratory Data Analysis (EDA):

Conduct exploratory analysis to understand the dataset's characteristics.
Visualize the distribution of home sales over the years.
Explore spatial patterns using maps, considering borough, neighborhood, or other relevant geographical features.
Identify trends, patterns, and outliers in the data.
Feature Engineering:

Extract relevant features from the dataset, such as sale date, location (latitude, longitude), property characteristics, etc.
Create additional features like month, year, season, and any other relevant temporal or spatial attributes.
Time Series Modeling:

Choose appropriate time series forecasting models such as ARIMA, SARIMA, Prophet, or LSTM based on the dataset's characteristics.
Split the data into training and testing sets, considering the chronological order of the sales data.
Train the selected model(s) using the training data.
Evaluate the model(s) using appropriate metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE).
Forecasting and Visualization:

Use the trained model(s) to make future sales predictions.
Visualize the actual vs. predicted sales over the test period to assess the model's accuracy.
Plot the forecasted sales for future years, providing insights into the expected market trends.
Incorporating Geospatial Information:

Utilize geospatial data from the NYCPluto shapefiles to visualize spatial distribution of sales.
Explore geographic patterns in sales using maps and spatial analysis techniques.
Correlate spatial features with sales trends to identify areas of high demand.
Conclusion and Recommendations:

Summarize the findings from the analysis and forecasting.
Provide insights into market trends, high-demand areas, and other relevant observations.
Offer recommendations for potential buyers, sellers, or real estate investors based on the forecasted trends.
Challenges and Limitations:

Discuss challenges faced during data preprocessing, modeling, or analysis.
Address limitations of the dataset and potential areas for improvement in future analyses.
Future Work:

Suggest potential areas for future research, such as incorporating additional external factors (e.g., economic indicators) for more accurate forecasts.
Propose enhancements to the model(s) for better performance and reliability.
Remember to document each step thoroughly, explain the rationale behind the choices made, and visualize the results effectively to communicate insights and findings clearly to your audience. Good luck with your project!