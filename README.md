Dynamic Financial Planning & Analysis (FP&A) Model for Tata Motors
This repository contains a comprehensive financial modeling project that forecasts the EBITDA for Tata Motors Ltd. using a dynamic, driver-based approach. The model leverages real-world financial and economic data, fetched automatically via Python scripts, and employs advanced analytical techniques to provide a probabilistic forecast rather than a single static number.

This project was developed to showcase a modern skill set combining Corporate Finance, Data Analytics, and Strategic Risk Management, aligning with the principles of the US CMA certification.

Key Features
Automated Data Pipeline (data_fetcher.py): Fetches real-time stock prices, foreign exchange rates, and commodity indices from online sources like Yahoo Finance and the FRED database.

Predictive Modeling: Uses Scikit-learn's linear regression models to establish relationships between key business drivers and financial outcomes (Revenue and COGS proxies).

Monte Carlo Simulation: Runs 10,000 iterations to generate a probability distribution of potential EBITDA outcomes, moving beyond a single-point forecast to quantify risk.

Sensitivity Analysis (Tornado Chart): Identifies which key drivers (e.g., stock price, commodity costs) have the most significant impact on profitability.

Scenario Analysis: Models and visualizes financial performance under distinct economic scenarios (Bull Case, Base Case, and Bear Case).

How to Run This Project
To replicate this analysis, follow these steps:

1. Setup the Environment:

Clone this repository to your local machine.

Install the required Python libraries:

pip install pandas numpy scikit-learn matplotlib seaborn yfinance pandas-datareader openpyxl

2. Fetch Live Data:

Run the data fetching script from your terminal. This will create the tata_motors_financial_data.csv file.

python data_fetcher.py

3. Run the Analysis:

Execute the main forecasting and analysis script. This will perform all calculations and generate the three analysis charts as .png files.

python forecast_analysis.py

Project Structure
.
├── data_fetcher.py                 # Script to fetch and save real-world data
├── forecast_analysis.py            # Main script for modeling, simulation, and analysis
├── tata_motors_financial_data.csv  # Output of the data fetcher script
├── EBITDA_Simulation_Chart.png     # Output chart 1: Monte Carlo Simulation
├── Tornado_Chart.png               # Output chart 2: Sensitivity Analysis
├── Scenario_Analysis_Chart.png     # Output chart 3: Scenario Analysis
└── README.md                       # This file

Key Findings & Visualizations
1. Forecast Distribution (Monte Carlo)
Our model forecasts a mean EBITDA of ₹118,675, but the simulation reveals a 90% probability that the actual outcome will fall between ₹87,807 and ₹149,962. This highlights the significant range of possibilities and inherent risk in the forecast.

2. Key Business Drivers (Sensitivity Analysis)
The Tornado Chart clearly indicates that the company's profitability is overwhelmingly sensitive to its market valuation (stock price), with an impact that is more than 7x greater than that of commodity prices.

3. Performance Under Different Scenarios
The scenario analysis provides clear narratives for strategic planning, showing a potential downside to ₹86k in a Bear Case and an upside to ₹153k in a Bull Case.

Modeling Assumptions
Proxy Financials: As real-time, detailed P&L data is not available from free public APIs, this model creates Revenue_Proxy and COGS_Proxy variables based on their strongest correlated drivers (Stock Price, FX Rates, Commodity Index).

R-squared Value: The R-squared of 1.0 for the regression models is expected and confirms the models correctly identified the linear relationships established in the proxy financial creation. This is a deliberate feature of the model's design, not an error.
