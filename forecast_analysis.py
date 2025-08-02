import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress the UserWarning from sklearn
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LinearRegression was fitted with feature names",
    category=UserWarning
)


# --- 1. LOAD AND PREPARE THE REAL DATA ---
try:
    # Load the real-world data you fetched
    df = pd.read_csv('tata_motors_financial_data.csv', index_col='Date', parse_dates=True)
    
    # FIX: The previous data fetcher saved columns as string representations of tuples.
    # This robustly cleans the column names by taking the first element.
    # e.g., "(Tata_Motors_Price, TATAMOTORS.NS)" becomes "Tata_Motors_Price"
    df.columns = [c.split(',')[0].strip("() '") for c in df.columns]

except FileNotFoundError:
    print("Error: 'tata_motors_financial_data.csv' not found.")
    print("Please run the data_fetcher.py script first to generate the data file.")
    exit()

# --- Create Proxy Financials ---
# Since we don't have P&L data, we create realistic proxies.
# This is a key modeling assumption you should mention in your presentation.
# We assume Revenue is correlated with stock price and FX rates.
# We assume COGS is correlated with commodity prices.
df['Revenue_Proxy'] = df['Tata_Motors_Price'] * 1000 + df['GBP_INR_Rate'] * 500
df['COGS_Proxy'] = df['Revenue_Proxy'] * 0.65 + df['Commodity_Index'] * 100

print("--- Data Loaded and Proxies Created ---")
print(df.head())
print("\n")


# --- 2. REGRESSION MODELING TO FIND KEY RELATIONSHIPS ---

# Model for Revenue
# We want to predict Revenue based on its key drivers
X_rev = df[['Tata_Motors_Price', 'GBP_INR_Rate']]
y_rev = df['Revenue_Proxy']
rev_model = LinearRegression()
rev_model.fit(X_rev, y_rev)
print(f"Revenue Model R-squared: {rev_model.score(X_rev, y_rev):.4f}")

# Model for Cost of Goods Sold (COGS)
# We want to predict COGS based on its key drivers
X_cogs = df[['Revenue_Proxy', 'Commodity_Index']]
y_cogs = df['COGS_Proxy']
cogs_model = LinearRegression()
cogs_model.fit(X_cogs, y_cogs)
print(f"COGS Model R-squared: {cogs_model.score(X_cogs, y_cogs):.4f}")
print("\n")


# --- 3. BASE CASE FORECAST FOR THE NEXT 12 MONTHS ---

# Forecast the drivers for the next year using simple growth assumptions
# A more complex model could use ARIMA or other time-series methods here.
last_known_price = df['Tata_Motors_Price'].iloc[-1]
last_known_fx = df['GBP_INR_Rate'].iloc[-1]
last_known_commodity = df['Commodity_Index'].iloc[-1]

# Assumptions for base case growth
price_growth = 1.05  # 5% growth in stock price
fx_growth = 1.02     # 2% appreciation in GBP/INR
comm_growth = 1.03   # 3% increase in commodity prices

# Forecasted driver values
base_price_forecast = last_known_price * price_growth
base_fx_forecast = last_known_fx * fx_growth
base_comm_forecast = last_known_commodity * comm_growth

# Helper function to calculate EBITDA based on input drivers.
def calculate_ebitda(price, fx, commodity, employee_pct=0.08, sga_pct=0.095):
    """Calculates EBITDA based on a set of driver inputs."""
    rev_df = pd.DataFrame([[price, fx]], columns=X_rev.columns)
    rev = rev_model.predict(rev_df)[0]
    
    cogs_df = pd.DataFrame([[rev, commodity]], columns=X_cogs.columns)
    cogs = cogs_model.predict(cogs_df)[0]
    
    ebitda = rev - cogs - (rev * employee_pct) - (rev * sga_pct)
    return ebitda

# Calculate Base Case EBITDA
base_ebitda = calculate_ebitda(base_price_forecast, base_fx_forecast, base_comm_forecast)
print("--- Base Case Forecast for Next Year ---")
print(f"Predicted EBITDA Proxy: {base_ebitda:,.0f}")
print("\n")


# --- 4. MONTE CARLO SIMULATION ---
N_SIMULATIONS = 10000
simulation_results = []

# Define distributions for uncertain variables
# Mean is the base case growth, scale (std dev) is based on historical volatility.
price_growth_std = df['Tata_Motors_Price'].pct_change().std()
comm_growth_std = df['Commodity_Index'].pct_change().std()

for i in range(N_SIMULATIONS):
    # Generate random growth rates for this simulation run
    sim_price_growth = np.random.normal(price_growth, price_growth_std)
    sim_comm_growth = np.random.normal(comm_growth, comm_growth_std)

    # Calculate simulated driver values
    sim_price = last_known_price * sim_price_growth
    sim_comm = last_known_commodity * sim_comm_growth

    # Calculate simulated EBITDA
    sim_ebitda = calculate_ebitda(sim_price, base_fx_forecast, sim_comm)
    simulation_results.append(sim_ebitda)


# --- 5. ANALYZE AND VISUALIZE SIMULATION RESULTS ---
results_series = pd.Series(simulation_results)

print("--- Monte Carlo Simulation Results (Next Year's EBITDA Proxy) ---")
print(f"Mean EBITDA: {results_series.mean():,.0f}")
print(f"Median EBITDA: {results_series.median():,.0f}")
print(f"5th Percentile: {results_series.quantile(0.05):,.0f}")
print(f"95th Percentile: {results_series.quantile(0.95):,.0f}")
print("\n")

# Create the histogram
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))
sns.histplot(results_series, kde=True, bins=50, ax=ax, color='#003366')

# Add lines for key statistics
ax.axvline(results_series.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {results_series.mean():,.0f}')
ax.axvline(results_series.quantile(0.05), color='black', linestyle=':', linewidth=2, label=f'5th Percentile: {results_series.quantile(0.05):,.0f}')
ax.axvline(results_series.quantile(0.95), color='black', linestyle=':', linewidth=2, label=f'95th Percentile: {results_series.quantile(0.95):,.0f}')

ax.set_title('Distribution of Simulated EBITDA Proxy (10,000 Iterations)', fontsize=16, fontweight='bold')
ax.set_xlabel('Simulated EBITDA Proxy', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=11)
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x)))
plt.tight_layout()

# Save the chart to a file to be used in your presentation
output_chart_filename = "EBITDA_Simulation_Chart_Real_Data.png"
fig.savefig(output_chart_filename, dpi=300)
print(f"Chart saved as '{output_chart_filename}'")
plt.show()


# --- 6. SENSITIVITY ANALYSIS (TORNADO CHART) ---
print("\n--- Performing Sensitivity Analysis ---")

# Define the range for sensitivity testing (e.g., +/- 10%)
sensitivity_factor = 0.10

# Test sensitivity for Tata Motors Price
price_low = base_price_forecast * (1 - sensitivity_factor)
price_high = base_price_forecast * (1 + sensitivity_factor)
ebitda_at_low_price = calculate_ebitda(price_low, base_fx_forecast, base_comm_forecast)
ebitda_at_high_price = calculate_ebitda(price_high, base_fx_forecast, base_comm_forecast)
price_sensitivity_range = ebitda_at_high_price - ebitda_at_low_price

# Test sensitivity for Commodity Index
comm_low = base_comm_forecast * (1 - sensitivity_factor)
comm_high = base_comm_forecast * (1 + sensitivity_factor)
ebitda_at_low_comm = calculate_ebitda(base_price_forecast, base_fx_forecast, comm_low)
ebitda_at_high_comm = calculate_ebitda(base_price_forecast, base_fx_forecast, comm_high)
comm_sensitivity_range = ebitda_at_high_comm - ebitda_at_low_comm

# Test sensitivity for FX Rate
fx_low = base_fx_forecast * (1 - sensitivity_factor)
fx_high = base_fx_forecast * (1 + sensitivity_factor)
ebitda_at_low_fx = calculate_ebitda(base_price_forecast, fx_low, base_comm_forecast)
ebitda_at_high_fx = calculate_ebitda(base_price_forecast, fx_high, base_comm_forecast)
fx_sensitivity_range = ebitda_at_high_fx - ebitda_at_low_fx


# Create the Tornado Chart
sensitivity_data = {
    'Variable': ['Tata Motors Price', 'Commodity Index', 'GBP/INR Rate'],
    'Sensitivity': [price_sensitivity_range, comm_sensitivity_range, fx_sensitivity_range]
}
sens_df = pd.DataFrame(sensitivity_data).sort_values(by='Sensitivity', ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(sens_df['Variable'], sens_df['Sensitivity'], color='#003366')
ax.set_title('Tornado Chart: Sensitivity of EBITDA to Key Drivers', fontsize=16, fontweight='bold')
ax.set_xlabel('Impact on EBITDA Proxy (Range of Outcomes)', fontsize=12)
ax.bar_label(bars, fmt='{:,.0f}', padding=5)
ax.axvline(0, color='black', linewidth=0.8) # Add a line at zero
plt.tight_layout()

# Save the Tornado chart
tornado_chart_filename = "Tornado_Chart.png"
fig.savefig(tornado_chart_filename, dpi=300)
print(f"Tornado chart saved as '{tornado_chart_filename}'")
plt.show()


# --- 7. SCENARIO ANALYSIS ---
print("\n--- Performing Scenario Analysis ---")

# Define scenario assumptions
# Bull Case: Strong market, lower commodity prices
bull_price_growth = 1.15  # 15% price growth
bull_comm_growth = 0.95   # 5% decrease in commodity prices

# Bear Case: Market downturn, high inflation
bear_price_growth = 0.90  # 10% price decrease
bear_comm_growth = 1.10   # 10% increase in commodity prices

# Calculate EBITDA for each scenario
bull_case_ebitda = calculate_ebitda(
    last_known_price * bull_price_growth,
    base_fx_forecast, # Keep FX stable for simplicity
    last_known_commodity * bull_comm_growth
)

bear_case_ebitda = calculate_ebitda(
    last_known_price * bear_price_growth,
    base_fx_forecast,
    last_known_commodity * bear_comm_growth
)

# Create the Scenario Analysis Chart
scenario_data = {
    'Scenario': ['Bear Case', 'Base Case', 'Bull Case'],
    'EBITDA': [bear_case_ebitda, base_ebitda, bull_case_ebitda]
}
scenario_df = pd.DataFrame(scenario_data)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#B22222', '#4682B4', '#228B22'] # Red, Blue, Green
bars = ax.bar(scenario_df['Scenario'], scenario_df['EBITDA'], color=colors)
ax.set_title('Scenario Analysis: Forecasted EBITDA Outcomes', fontsize=16, fontweight='bold')
ax.set_ylabel('EBITDA Proxy', fontsize=12)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x)))
ax.bar_label(bars, fmt='{:,.0f}', padding=3)
plt.tight_layout()

# Save the Scenario chart
scenario_chart_filename = "Scenario_Analysis_Chart.png"
fig.savefig(scenario_chart_filename, dpi=300)
print(f"Scenario analysis chart saved as '{scenario_chart_filename}'")
plt.show()
