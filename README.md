# US Trade Balance Forecasting

In this project, we forecast the **U.S. International Trade Balance in Goods and Services**. International Trade Balance in Goods and Services attempts to
accurately measure the trade balance of the United States, which is the difference between imports and exports measured in the millions of US dollars ($).

## Data

The original publisher of the data we use is the [Bureau of Economic Analysis (BEA)](https://www.bea.gov/). However, we obtained the data from the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/) using the FRED code `BOPGSTB`.

This data is present in `BOPGSTB.csv`.

## Evaluation Criteria

For the purpose of comparing different models we use for our data, the metric we use is the `Akaike Information Criterion (AIC)` to choose the best forecasting model. Since we are just beginning to work out a model for the Trade Balance time series, we believe that all models we come up with are just approximations of the true model, and **our goal here is to find the best approximation, rather than finding the true model**. Moreover, we do
not have access to out-of-sample data. These are the reasons for us to choose the `AIC` criterion over others like the `Bayesian Information Criterion (BIC)`.

## Repository API

The main files and directory of this repository are as follows:
- 
- 
- 

## Model

We choose 
