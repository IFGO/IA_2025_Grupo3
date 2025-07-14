# -*- coding: utf-8 -*-
"""
Refactored script for cryptocurrency analysis.

This script loads historical data for several cryptocurrencies, calculates
summary and dispersion statistics, and generates various plots to visualize
the data, such as boxplots, histograms, and time series graphs. All plots
are saved to a 'figures' directory.
"""

import os
import urllib.error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# The 'display' function is part of IPython and works in environments like Jupyter Notebooks.
from IPython.display import display

def setup_environment():
    """Creates the 'figures' directory if it doesn't exist."""
    if not os.path.exists('figures'):
        os.makedirs('figures')
        print("Directory 'figures' created.")

def load_crypto_data(crypto_urls):
    """
    Loads cryptocurrency data from local CSV files into a dictionary of DataFrames.

    Args:
        crypto_urls (list): A list of dictionaries, each containing the name 
                            and local path of a cryptocurrency's CSV file.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary of DataFrames, with coin tickers as keys.
            - list: A list of coin tickers.
    """
    dfs = {}
    coins = []
    print("Loading data...")
    for crypto_data in crypto_urls:
        name = crypto_data['name']
        url = crypto_data['url']
        try:
            df_name = name.split(' ')[0]  # Extract ticker (e.g., 'BTC')
            coins.append(df_name)
            # Assumes CSVs are in a local directory specified by the path in `url`
            dfs[df_name] = pd.read_csv(url, skiprows=1)
            print(f"DataFrame '{df_name}' created successfully from {url}.")
        except FileNotFoundError:
            print(f"Error: The file for {name} was not found at the path: {url}.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {name} from {url}: {e}")
    return dfs, coins

def highlight_rows(row):
    """Helper function to apply alternating row colors to a DataFrame style."""
    return ['background-color: #6495ED;' if row.name % 2 == 0 else '' for _ in row]

def dataframe_to_csv(file_name: str, df: pd.DataFrame):
    """
    Exports the given dataframe to csv file
    
    Args:
        df (pd.DataFrame): The DataFrame to export.
    """
    print(file_name)
    df.to_csv(f"./tables/{file_name}.csv", encoding='utf-8')
    return df

def plot_closing_price_boxplots(dfs, coins):
    """
    Generates and saves boxplots of closing prices by year for each cryptocurrency.

    Args:
        dfs (dict): A dictionary of DataFrames for each cryptocurrency.
        coins (list): A list of cryptocurrency names.
    """
    print("\nGenerating closing price boxplots...")
    num_coins = len(coins)
    num_cols = 2
    num_rows = (num_coins + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), dpi=150)
    axes = axes.flatten()

    for i, coin in enumerate(coins):
        df = dfs[coin].copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        years = sorted(df['year'].unique())
        # Filter out years with no data to avoid errors
        data_to_plot = [df[df['year'] == year]['close'] for year in years if not df[df['year'] == year].empty]
        
        if not data_to_plot:
            continue

        ax = axes[i]
        ax.boxplot(data_to_plot, patch_artist=True, tick_labels=years)
        ax.set_title(f'Closing Prices - {coin}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Closing Value (USD)')

    # Hide any unused subplots
    for j in range(num_coins, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_path = 'figures/closing_price_boxplots.png'
    plt.savefig(save_path)
    print(f"Saved boxplots to {save_path}")
    

def plot_closing_price_histograms(dfs, coins):
    """
    Generates and saves histograms of closing prices for each cryptocurrency.

    Args:
        dfs (dict): A dictionary of DataFrames for each cryptocurrency.
        coins (list): A list of cryptocurrency names.
    """
    print("\nGenerating closing price histograms...")
    n_coins = len(coins)
    n_cols = 2
    n_rows = (n_coins + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5), dpi=150)
    axes = axes.flatten()

    for i, coin in enumerate(coins):
        if coin in dfs and not dfs[coin].empty and 'close' in dfs[coin].columns:
            sns.histplot(data=dfs[coin], x='close', ax=axes[i], kde=True)
            axes[i].set_title(f'Closing Price Histogram for {coin}')
            axes[i].set_xlabel('Closing Price')
            axes[i].set_ylabel('Frequency')
        else:
            print(f"No valid data found for cryptocurrency {coin}.")
            fig.delaxes(axes[i])

    for j in range(n_coins, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_path = 'figures/closing_price_histograms.png'
    plt.savefig(save_path)
    print(f"Saved histograms to {save_path}")
    

def plot_historical_closing_prices(dfs, coins):
    """
    Generates and saves time series plots of historical closing prices.

    Args:
        dfs (dict): A dictionary of DataFrames for each cryptocurrency.
        coins (list): A list of cryptocurrency names.
    """
    print("\nGenerating historical closing price plots...")
    n_coins = len(coins)
    n_cols = 2
    n_rows = (n_coins + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5), dpi=150)
    axes = axes.flatten()

    for i, coin in enumerate(coins):
        if coin in dfs and not dfs[coin].empty and 'close' in dfs[coin].columns:
            df = dfs[coin].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by='date')

            df['mean'] = df['close'].rolling(window=7).mean()
            df['median'] = df['close'].rolling(window=7).median()
            df['mode'] = df['close'].rolling(window=7).apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

            ax = axes[i]
            ax.plot(df['date'], df['close'], label='Fechamento', alpha=0.6)
            ax.plot(df['date'], df['mean'], label='Média (7d)', linestyle='--')
            ax.plot(df['date'], df['median'], label='Mediana (7d)', linestyle='--')
            ax.plot(df['date'], df['mode'], label='Moda (7d)', linestyle='--')

            ax.set_title(f'Preço de Fechamento de {coin} (com Média, Mediana e Moda)')
            ax.set_xlabel('Data')
            ax.set_ylabel('Preço')
            ax.legend()
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)
        else:
            fig.delaxes(axes[i])

    for j in range(n_coins, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_path = 'figures/historical_closing_prices.png'
    plt.savefig(save_path)
    print(f"Saved historical plots to {save_path}")
    

def display_summary_statistics(dfs, coins):
    """
    Calculates and displays summary statistics for each cryptocurrency.

    Args:
        dfs (dict): A dictionary of DataFrames for each cryptocurrency.
        coins (list): A list of cryptocurrency names.
    """
    print("\n--- Summary Statistics ---")
    for coin in coins:
        title = f"Summary statistics for - {coin}"
        print(f"\n{title}:")

        coin_data = dfs[coin].drop(columns=['unix', 'date', 'symbol'], errors='ignore')
        summary = {'Metric': [], 'Mean': [], 'Median': [], 'Mode': [], 'Q1 (25%)': [], 'Q2 (50%)': [], 'Q3 (75%)': []}

        for col in coin_data.columns:
            summary['Metric'].append(col)
            summary['Mean'].append(coin_data[col].mean())
            summary['Median'].append(coin_data[col].median())
            summary['Mode'].append(coin_data[col].mode().iloc[0] if not coin_data[col].mode().empty else np.nan)
            quantiles = coin_data[col].quantile([0.25, 0.5, 0.75])
            summary['Q1 (25%)'].append(quantiles[0.25])
            summary['Q2 (50%)'].append(quantiles[0.5])
            summary['Q3 (75%)'].append(quantiles[0.75])

        df_summary = pd.DataFrame(summary)
        pd.set_option('display.float_format', '{:.8f}'.format)
        
        # Note: The 'display' function is used here for its rich output in notebooks.
        # In a standard Python script, you would use 'print(df_summary)'.
        print(df_summary)
        dataframe_to_csv(title, df_summary)

def display_dispersion_measures(dfs, coins):
    """
    Calculates and displays measures of dispersion for each cryptocurrency.

    Args:
        dfs (dict): A dictionary of DataFrames for each cryptocurrency.
        coins (list): A list of cryptocurrency names.
    """
    print("\n--- Dispersion Measures ---")
    vari_summary = {'Coin': [], 'Coef.Variation': []}
    for coin in coins:
        title = f"Dispersion measures for - {coin}"
        print(f"{title}:")
        coin_data = dfs[coin].drop(columns=['unix', 'date', 'symbol'], errors='ignore')
        dispersion = {'Metric': [], 'Variance': [], 'Std. Deviation': [], 'Range': [], 'Interquartile Range': [], 'Coef. of Variation': []}

        for col in coin_data.columns:
            dispersion['Metric'].append(col)
            dispersion['Variance'].append(coin_data[col].var(ddof=1))
            dispersion['Std. Deviation'].append(coin_data[col].std(ddof=1))
            dispersion['Range'].append(coin_data[col].max() - coin_data[col].min())
            q3 = coin_data[col].quantile(0.75)
            q1 = coin_data[col].quantile(0.25)
            dispersion['Interquartile Range'].append(q3 - q1)
            coef_var = coin_data[col].std() / coin_data[col].mean()
            dispersion['Coef. of Variation'].append(coef_var)
            if col == 'close':
                vari_summary['Coin'].append(coin)
                vari_summary['Coef.Variation'].append(coef_var)

        df_dispersion = pd.DataFrame(dispersion)
        pd.set_option('display.float_format', '{:.8f}'.format)    
        print(dataframe_to_csv(title, df_dispersion))    
    
    df_coef_var = pd.DataFrame(vari_summary)
    print("\n--- Comparison of Coefficient of Variation (Closing Price) ---")
    pd.set_option('display.float_format', '{:.8f}'.format)
    print(dataframe_to_csv('Coef.Variation', df_coef_var.sort_values(by='Coef.Variation', ascending=False)))
    print("\n" + "-"*80)

def analyze_daily_volatility(dfs, coins):
    """
    Analyzes and plots the daily volatility for each cryptocurrency.

    Args:
        dfs (dict): A dictionary of DataFrames for each cryptocurrency.
        coins (list): A list of cryptocurrency names.
    """
    print("\n--- Daily Volatility Analysis ---")

    for coin in coins:
        if coin in dfs and not dfs[coin].empty and all(c in dfs[coin].columns for c in ['high', 'low', 'close']):
            df_coin = dfs[coin].copy()
            df_coin['daily_range'] = df_coin['high'] - df_coin['low']

            print(f"\nDaily and Overall Variation Measures for {coin}:")

            mean_daily_range = df_coin['daily_range'].mean()
            median_daily_range = df_coin['daily_range'].median()
            std_daily_range = df_coin['daily_range'].std()

            print(f"  Daily Range (High - Low):")
            print(f"    Mean: {mean_daily_range:.8f}")
            print(f"    Median: {median_daily_range:.8f}")
            print(f"    Standard Deviation: {std_daily_range:.8f}")
            
            q3_close = df_coin['close'].quantile(0.75)
            q1_close = df_coin['close'].quantile(0.25)
            dist_interquart_close = q3_close - q1_close
            range_close = df_coin['close'].max() - df_coin['close'].min()

            print(f"\n  Overall Closing Price Variation:")
            print(f"    Range: {range_close:.8f}")
            print(f"    Interquartile Range: {dist_interquart_close:.8f}")

            # Plotting the daily volatility
            plt.figure(figsize=(12, 6), dpi=150)
            plt.plot(pd.to_datetime(df_coin['date']), df_coin['daily_range'])
            plt.title(f'Daily Volatility (High - Low) of {coin} Over Time')
            plt.xlabel('Date')
            plt.ylabel('Daily Price Range (USD)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            
            save_path = f'figures/daily_volatility_{coin}.png'
            plt.savefig(save_path)
            print(f"Saved daily volatility plot to {save_path}")
            
        else:
            print(f"\nIncomplete or non-existent data for {coin} for volatility analysis.")

def run_statistical():
    """
    Main function to execute the cryptocurrency analysis workflow.
    """
    # Create the 'figures' directory to save plots
    setup_environment()

    # Define the list of cryptocurrencies and their local data file paths
    # IMPORTANT: Create a 'data' folder and place your CSV files there.
    crypto_urls = [
        {'name':'BTC (Bitcoin)','url':'./data/Bitfinex_BTCUSD_d.csv'},
        {'name':'ETH (Ethereum)','url':'./data/Bitfinex_ETHUSD_d.csv'},
        {'name':'XRP (Ripple)','url':'./data/Bitfinex_XRPUSD_d.csv'},
        {'name':'ADA (Cardano)','url':'./data/Bitfinex_ADAUSD_d.csv'},
        {'name':'DAI (DAI)','url':'./data/Bitfinex_DAIUSD_d.csv'},
        {'name':'LTC (Litecoin)','url':'./data/Bitfinex_LTCUSD_d.csv'},
        {'name':'DOT (Polkadot)','url':'./data/Bitfinex_DOTUSD_d.csv'},
        {'name':'SOL (Solana)','url':'./data/Bitfinex_SOLUSD_d.csv'},
        {'name':'TRX (Tron)','url':'./data/Bitfinex_TRXUSD_d.csv'},
        {'name':'XMR (Monero)','url':'./data/Bitfinex_XMRUSD_d.csv'}
    ]

    # Load the data
    dfs, coins = load_crypto_data(crypto_urls)
    
    # Exit if no data was loaded
    if not dfs:
        print("\nNo data was loaded. Exiting the script.")
        return

    # Run the analyses
    display_summary_statistics(dfs, coins)
    display_dispersion_measures(dfs, coins)
    
    # Generate and save plots
    plot_closing_price_boxplots(dfs, coins)
    plot_closing_price_histograms(dfs, coins)
    plot_historical_closing_prices(dfs, coins)
    
    # Analyze and plot daily volatility
    analyze_daily_volatility(dfs, coins)
    
    print("\nAnalysis complete.")