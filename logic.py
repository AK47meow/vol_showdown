# logic.py = calculations and helper funcs
import scipy
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
# Import streamlit here because some functions use st.error/st.warning directly
import streamlit as st

# These imports are for plotting in other contexts, okay to keep but not strictly needed for this Streamlit app
from mpl_toolkits.mplot3d import Axes3D  # Restored this import
from scipy.optimize import newton
import seaborn as sns  # RESTORED: import seaborn as sns

from math import exp, sqrt, log
from datetime import date, timedelta, datetime


def calculate_sigma(stock_data):
    df = stock_data.copy()
    # Ensure there's enough data for log return calculation
    if len(df) < 2:
        return 0.2  # Default sigma if not enough historical data
    df["Log Return"] = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if df["Log Return"].empty:
        return 0.2  # Default sigma if log returns are all NaN
    sigma = np.std(df["Log Return"]) * np.sqrt(252)
    return sigma

def compute_annualized_T(expiry_str, today=datetime.now()):
    """
    Computes the time to maturity in years from an expiry date string.
    Ensures T is at least 0.01 to avoid issues with division by zero or log(0).
    """
    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
    T_days = (expiry_date - today).days
    return max(T_days / 365, 0.01) if T_days > 0 else None


def calculate_black_scholes(S, K, T, r, sigma, option_type):
    S = float(S)
    K = float(K)
    T = float(T)
    sigma = float(sigma)
    r = float(r)

    # Handle cases where T or sigma might be zero/near-zero which causes division by zero
    if T <= 0 or sigma <= 1e-6:  # sigma should not be 0
        return 0.0  # Option price is 0 if no time or no volatility

    # Calculate d1 and d2 parameters, handling potential invalid inputs
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
        d2 = d1 - (sigma * np.sqrt(T))

    # Handle cases where d1 or d2 might become inf/nan due to extreme inputs
    if np.isinf(d1) or np.isnan(d1) or np.isinf(d2) or np.isnan(d2):
        return 0.0  # Return 0 for invalid inputs

    bs_price = 0.0  # Initialize bs_price to ensure it's always defined
    try:
        if option_type == "c":
            bs_price = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
        elif option_type == "p":
            bs_price = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))
        else:
            return 0.0  # Return 0 for invalid option type
    except Exception:
        return 0.0

    return bs_price


def calculate_vega(S, K, T, r, sigma, option_type):
    """Calculates Vega (sensitivity of option price to volatility)"""
    S = float(S)
    K = float(K)
    T = float(T)
    sigma = float(sigma)
    r = float(r)

    if T <= 0 or sigma <= 1e-6:  # Handle non-positive time or zero volatility
        return 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))

    if np.isinf(d1) or np.isnan(d1):  # Handle cases where d1 is invalid
        return 0.0

    try:
        # Vega is the same for calls and puts
        vega = S * norm.pdf(d1, 0, 1) * np.sqrt(T)
        return vega
    except Exception:
        return 0.0


def implied_vol_newton(C, S, K, r, T, tol, option_type):
    def inflexion_point(S_val, K_val, T_val, r_val):  # Renamed args to avoid shadowing
        if T_val <= 0 or K_val <= 0:  # Handle cases that would cause issues with log/division
            return 0.2
        try:
            m = S_val / (K_val * np.exp(-r_val * T_val))
            # Handle cases where m <= 0, which would cause log(0) or log(negative)
            if m <= 0:
                return 0.2
            return np.sqrt(2 * np.abs(np.log(m)) / T_val)
        except RuntimeWarning:
            return 0.2  # Fallback to a common guess

    guess = inflexion_point(S, K, T, r)
    if guess is None or np.isnan(guess) or guess <= 0:  # Ensure guess is a valid positive number
        guess = 0.5  # Fallback to a common volatility if initial guess is problematic

    max_iter = 300
    for _ in range(max_iter):
        p = calculate_black_scholes(S, K, T, r, guess, option_type)
        v = calculate_vega(S, K, T, r, guess, option_type)

        # Avoid division by zero for tiny Vega
        if abs(v) < 1e-6:
            return None

        error = (p - C) / v
        # ensure scalar
        try:
            error = float(error)
        except Exception:
            pass

        guess = guess - error
        if abs(error) < tol:
            return float(guess)

    return None  # No convergence


def implied_vol_brentq(C, S, K, r, T, option_type):
    def objective(sigma):
        # Ensure sigma is positive
        if sigma <= 0:
            return C
        return calculate_black_scholes(S, K, T, r, sigma, option_type) - C

    lower_bound = 1e-6  # Very small positive volatility
    upper_bound = 5.0   # Very high volatility

    try:
        price_at_low_vol = calculate_black_scholes(S, K, T, r, lower_bound, option_type)
        price_at_high_vol = calculate_black_scholes(S, K, T, r, upper_bound, option_type)

        if not (min(price_at_low_vol, price_at_high_vol) <= C <= max(price_at_low_vol, price_at_high_vol)):
            return np.nan  # No sign change => no root in [lower, upper]

        return brentq(objective, lower_bound, upper_bound, maxiter=1000)
    except (ValueError, RuntimeError):
        return np.nan


# run_analysis = my brain, Import data and plot
@st.cache_data(ttl=1800, max_entries=8, show_spinner=False)
def run_analysis(ticker, option_type="c"):
    """
    Fetches option chain data, calculates various option parameters including implied volatility,
    and prepares data for 3D IVS plotting.

    NOTE: No 'impliedVolatility_calculated' column is created.
          The IV surface prefers 'volatility_brentq' and falls back to 'volatility_newton' if needed.
    """
    try:
        stock = yf.Ticker(ticker)
        # Current stock price
        stock_info = stock.history(period="1d")
        if stock_info.empty:
            st.error(f"Could not fetch historical stock price for {ticker}. Please check the ticker symbol or internet connection.")
            return None, None, None, None
        S = stock_info["Close"].iloc[-1]

        # Historical volatility (1y)
        hist_data = stock.history(period="1y")
        if hist_data.empty or len(hist_data) < 2:
            st.warning(f"Not enough historical data for {ticker} to calculate sigma. Using a default volatility of 0.2.")
            sigma = 0.2
        else:
            sigma = calculate_sigma(hist_data)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                st.warning(f"Calculated historical sigma for {ticker} is invalid. Using a default volatility of 0.2.")
                sigma = 0.2

        # Risk-free rate from ^TNX (10y)
        tnx_data = yf.download("^TNX", period="1d")
        if tnx_data.empty:
            st.warning("Could not fetch ^TNX (risk-free rate) data. Using a default risk-free rate of 0.01.")
            r = 0.01
        else:
            r = (tnx_data["Close"].iloc[-1] / 100).item()
            if r is None or np.isnan(r) or r < 0:
                st.warning(f"Calculated risk-free rate for ^TNX is invalid. Using a default risk-free rate of 0.01.")
                r = 0.01

        contracts_list = []

        # Iterate expirations
        for expiry in stock.options:
            T = compute_annualized_T(expiry)
            if T is None:
                continue

            try:
                chain = stock.option_chain(expiry)
                if option_type == "c":
                    options_df = chain.calls
                elif option_type == "p":
                    options_df = chain.puts
                else:
                    continue

                options_df = options_df.dropna(subset=["strike", "lastPrice"])
                options_df = options_df[options_df["lastPrice"] > 0.0001]
                if options_df.empty:
                    continue

                options_df["expirationDate"] = expiry
                options_df["S"] = S
                options_df["K"] = options_df["strike"]
                options_df["C"] = options_df["lastPrice"]
                options_df["T"] = T
                options_df["r"] = r
                options_df["sigma_hist"] = sigma
                options_df["type"] = option_type
                contracts_list.append(options_df)

            except Exception:
                continue

        if not contracts_list:
            st.warning("No valid option contracts found for the selected ticker and type. Try another ticker or check internet connection.")
            return None, None, None, None

        contracts = pd.concat(contracts_list, ignore_index=True)

        # Enrichment
        columns_to_drop = ["bid", "ask", "change", "percentChange", "volume", "openInterest", "contractSize", "currency"]
        contracts.drop(columns=columns_to_drop, inplace=True, errors="ignore")

        # Moneyness
        contracts["moneyness"] = contracts["strike"].astype(float) / S

        # yfinance IV (if present)
        if "impliedVolatility" in contracts.columns:
            contracts["impliedVolatility"] = contracts["impliedVolatility"].astype(float)
        else:
            contracts["impliedVolatility"] = np.nan

        # BSM price & Vega with sigma_hist
        contracts["blackScholes_model_price"] = contracts.apply(
            lambda c: calculate_black_scholes(c["S"], c["K"], c["T"], c["r"], c["sigma_hist"], c["type"]), axis=1
        )
        contracts["vega"] = contracts.apply(
            lambda c: calculate_vega(c["S"], c["K"], c["T"], c["r"], c["sigma_hist"], c["type"]), axis=1
        )

        # Implied vols: Newton & BrentQ (kept separate for pedagogy)
        contracts["volatility_newton"] = contracts.apply(
            lambda c: implied_vol_newton(c["C"], c["S"], c["K"], c["r"], c["T"], 0.0001, c["type"]), axis=1
        )
        contracts["volatility_brentq"] = contracts.apply(
            lambda c: implied_vol_brentq(c["C"], c["S"], c["K"], c["r"], c["T"], c["type"]), axis=1
        )

        # ---------- IV Surface (no combined column) ----------
        # Prefer BrentQ; if insufficient points, fallback to Newton
        brentq_valid = contracts.dropna(subset=["moneyness", "T", "volatility_brentq"]).copy()
        brentq_valid = brentq_valid[
            (brentq_valid["volatility_brentq"] > 0.001) &
            (brentq_valid["volatility_brentq"] < 5.0)
        ]

        if not brentq_valid.empty:
            subset_for_plot = brentq_valid
            z_vals = subset_for_plot["volatility_brentq"].values
        else:
            newton_valid = contracts.dropna(subset=["moneyness", "T", "volatility_newton"]).copy()
            newton_valid = newton_valid[
                (newton_valid["volatility_newton"] > 0.001) &
                (newton_valid["volatility_newton"] < 5.0)
            ]
            if not newton_valid.empty:
                subset_for_plot = newton_valid
                z_vals = subset_for_plot["volatility_newton"].values
            else:
                st.info("Not enough valid data points to build a meaningful IVS. Try a different ticker or expiry.")
                return contracts, None, None, None

        x = subset_for_plot["moneyness"].values
        y = subset_for_plot["T"].values
        z = z_vals

        # Create grid & interpolate
        xi = np.linspace(x.min(), x.max(), 100) if x.size > 1 else np.array([x.min(), x.min() + 0.1])
        yi = np.linspace(y.min(), y.max(), 100) if y.size > 1 else np.array([y.min(), y.min() + 0.1])
        xi, yi = np.meshgrid(xi, yi)

        if len(x) >= 4 and len(np.unique(x)) >= 2 and len(np.unique(y)) >= 2:
            zi = griddata((x, y), z, (xi, yi), method="cubic")
        elif len(x) >= 1:
            st.warning("Not enough data points for cubic interpolation, falling back to linear/nearest interpolation for IVS.")
            zi = griddata((x, y), z, (xi, yi), method="linear", fill_value=np.nan)
            if np.all(np.isnan(zi)):
                zi = griddata((x, y), z, (xi, yi), method="nearest")
        else:
            zi = None

        return contracts, xi, yi, zi

    except Exception as e:
        st.error(f"An unexpected error occurred during analysis: {e}. Please check the ticker symbol, option type, or your internet connection.")
        return None, None, None, None