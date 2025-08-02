import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime

def fetch_and_save_data(start_date, end_date, output_filename="tata_motors_financial_data.csv"):
    """
    Fetches financial, commodity, and FX data for Tata Motors, cleans it,
    and saves it to a CSV file. This version is updated for API stability.

    Args:
        start_date (datetime.date): The start date for data fetching.
        end_date (datetime.date): The end date for data fetching.
        output_filename (str): The name of the output CSV file.
    """
    print("--- Starting Data Fetching Process ---")

    # --- 1. Define Tickers and Series IDs ---
    stock_ticker = 'TATAMOTORS.NS'
    fx_ticker = 'GBPINR=X'
    commodity_series_id = 'PALLFNFINDEXM' # Global Price Index of All Commodities

    data_sources = {}

    # --- 2. Fetch Data with Individual Error Handling ---
    # Fetch Stock Data
    try:
        print(f"Fetching Tata Motors stock data ({stock_ticker})...")
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date, interval='1mo', auto_adjust=True, progress=False)
        if not stock_data.empty and 'Close' in stock_data.columns:
            data_sources['stock'] = stock_data[['Close']].rename(columns={'Close': 'Tata_Motors_Price'})
        else:
            print(f"Warning: No stock data found for {stock_ticker}.")
    except Exception as e:
        print(f"An error occurred fetching stock data for {stock_ticker}: {e}")

    # Fetch FX Data
    try:
        print(f"Fetching GBP/INR FX data ({fx_ticker})...")
        fx_data = yf.download(fx_ticker, start=start_date, end=end_date, interval='1mo', auto_adjust=True, progress=False)
        if not fx_data.empty and 'Close' in fx_data.columns:
            data_sources['fx'] = fx_data[['Close']].rename(columns={'Close': 'GBP_INR_Rate'})
        else:
            print(f"Warning: No FX data found for {fx_ticker}.")
    except Exception as e:
        print(f"An error occurred fetching FX data for {fx_ticker}: {e}")

    # Fetch Commodity Data from FRED
    try:
        print("Fetching Commodity data from FRED...")
        commodity_data = web.DataReader(commodity_series_id, 'fred', start_date, end_date)
        if not commodity_data.empty:
            comm_series = commodity_data[commodity_series_id]
            data_sources['commodities'] = comm_series.to_frame(name='Commodity_Index')
        else:
            print(f"Warning: No commodity data found for series {commodity_series_id}.")
    except Exception as e:
        print(f"An error occurred during FRED data fetching: {e}")

    if not data_sources:
        print("Could not fetch any data. Please check your connection and ticker symbols. Exiting.")
        return

    # --- 3. Clean and Combine Data ---
    # FIX: Use pd.concat for a more robust merge of all data sources.
    print("Cleaning and combining data...")

    # Combine all successfully fetched dataframes into a single one
    final_df = pd.concat(data_sources.values(), axis=1, join='outer')

    # Resample to monthly start and forward-fill to handle non-trading days/lags
    final_df = final_df.resample('MS').ffill()
    final_df.index.name = 'Date'

    # Drop any rows where crucial data might still be missing after forward-filling
    final_df.dropna(inplace=True)

    print("Data processing complete.")

    # --- 4. Save to CSV ---
    if final_df.empty:
        print("Final DataFrame is empty. No data to save.")
        return

    try:
        final_df.to_csv(output_filename)
        print(f"\nSuccessfully saved data to '{output_filename}'")
        print("--- Data Preview ---")
        print(final_df.head())
        print("--------------------")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == '__main__':
    # Define the time period for which to fetch data
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=7*365)

    fetch_and_save_data(start_date, end_date)
