from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib.pyplot as plt
import os
import yfinance as yf
import scipy.stats as stats

# Set Matplotlib backend to 'Agg' to prevent GUI-related errors
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

def fetch_crypto_data(ticker):
    crypto = yf.Ticker(ticker)
    hist = crypto.history(period="1y")  # Fetch historical data for the past year
    latest_close = hist['Close'].iloc[-1]  # Get the latest close price
    returns = hist['Close'].pct_change().dropna()
    sigma = np.std(returns) * np.sqrt(252)  # Annualized volatility
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns) + 3  # Excess kurtosis to normal kurtosis
    return latest_close, sigma, skewness, kurtosis, hist['Close'].values

@app.route('/', methods=['GET', 'POST'])
def index():
    default_ticker = 'BTC-USD'
    years = 1  # default time period
    risk_free_rate = 0.05  # default risk-free rate

    if request.method == 'POST':
        ticker = request.form.get('ticker', default_ticker)
        latest_close, sigma, skewness, kurtosis, historical_prices = fetch_crypto_data(ticker)

        if 'update_ticker' in request.form:
            return render_template('index.html', ticker=ticker, Close=latest_close, years=years, rate=risk_free_rate, sigma=sigma, skewness=skewness, kurtosis=kurtosis, 
                                   K=latest_close, T=years, r=risk_free_rate, num_simulations=10000, num_steps=252, 
                                   call_option_price=None, put_option_price=None, plot_url=None)

        try:
            K = float(request.form['K'])
            T = float(request.form['T'])
            r = float(request.form['r'])
            sigma = float(request.form['sigma'])
            skewness = float(request.form['skewness'])
            kurtosis = float(request.form['kurtosis'])
            num_simulations = int(request.form['num_simulations'])
            num_steps = int(request.form['num_steps'])
        except KeyError as e:
            return f"Missing form field: {e}", 400
        except Exception as e:
            return f"An error occurred: {e}", 400

        dt = T / num_steps
        P = np.zeros((num_simulations, num_steps + 1))
        P[:, 0] = latest_close

        for t in range(1, num_steps + 1):
            Z = np.random.standard_normal(num_simulations)
            adjusted_Z = Z + (skewness / 6) * (Z**2 - 1) + (kurtosis / 24) * (Z**3 - 3*Z) - (skewness**2 / 36) * (2*Z**3 - 5*Z)
            P[:, t] = P[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * adjusted_Z)

        call_payoff = np.maximum(P[:, -1] - K, 0)
        put_payoff = np.maximum(K - P[:, -1], 0)

        call_option_price = np.exp(-r * T) * np.mean(call_payoff)
        put_option_price = np.exp(-r * T) * np.mean(put_payoff)

        # Determine y-axis limits
        y_max = np.percentile(P[:, -1], 99)  # Set y-axis limit to 99th percentile of simulated prices
        y_min = np.percentile(P[:, -1], 1)   # Set y-axis limit to 1st percentile of simulated prices

        plt.figure(figsize=(10, 6))
        plt.plot(P.T, color='grey', alpha=0.1)
        plt.title(f'Monte Carlo Simulation: {num_simulations} Paths of {ticker}')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.ylim(y_min, y_max)  # Set y-axis limits dynamically
        plot_path = os.path.join('static', 'plot.png')
        plt.savefig(plot_path)
        plt.close()

        return render_template('index.html', ticker=ticker, Close=latest_close, years=T, rate=r, sigma=sigma, skewness=skewness, kurtosis=kurtosis,
                               K=K, T=T, r=r, num_simulations=num_simulations, num_steps=num_steps, 
                               call_option_price=f"{call_option_price:.2f}", put_option_price=f"{put_option_price:.2f}", 
                               plot_url=url_for('static', filename='plot.png'))
    
    # Default values for the form inputs
    latest_close, sigma, skewness, kurtosis, _ = fetch_crypto_data(default_ticker)
    return render_template('index.html', ticker=default_ticker, Close=latest_close, years=years, rate=risk_free_rate, sigma=sigma, skewness=skewness, kurtosis=kurtosis, 
                           K=latest_close, T=years, r=risk_free_rate, num_simulations=10000, num_steps=252, 
                           call_option_price=None, put_option_price=None, plot_url=None)

if __name__ == '__main__':
    app.run(debug=True)
