from pot import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_mock_data(size=1000, extreme_events=15, extreme_magnitude=5):
    data = np.random.randn(size) * 0.5 + 1
    
    # Randomly add some outlier
    extreme_indices = np.random.randint(0, size, size=extreme_events)
    data[extreme_indices] += np.random.rand(extreme_events) * extreme_magnitude + extreme_magnitude / 2
    
    return pd.Series(data)

# ----- Mock data analysis ----- #
mock_data = generate_mock_data(size=1200, extreme_events=40, extreme_magnitude=8)
threshold = mock_data.quantile(0.95)
exceedances, excesses, exceedance_indices = analyze_pot(mock_data, threshold)

print("--- Peaks-Over-Threshold (POT) Results ---")
print(f"Data points amount: {len(mock_data)}")
print(f"95th percentile as threshold: {threshold:.4f}")
if len(exceedances) > 0:
    print(f"The amount of data over threshold: {len(exceedances)}")
    print(f"Max Excess: {excesses.max():.4f}")
    print(f"Mean Excess: {excesses.mean():.4f}")
else:
    print("No data is over threshold")

# ----- visualization ----- #
if len(exceedances) > 0:
    # ----- Set the figure ----- #
    plt.figure(figsize=(15, 10))
    plt.style.use('seaborn-v0_8-whitegrid')

    # ----- Original data and over threshold points ----- #
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(mock_data, label='mock data', color='royalblue', alpha=0.7)
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'threshold ({threshold:.2f})')
    ax1.scatter(exceedance_indices, exceedances, color='orangered', zorder=5, label='Exceedances')
    ax1.set_title('Time Series data and over threshold rare event', fontsize=16)
    ax1.set_xlabel('time/index')
    ax1.set_ylabel('value')
    ax1.legend()

    # ----- The ECDF of excessdances ----- #
    ax2 = plt.subplot(2, 1, 2)

    # ----- Compute ECDF ----- #
    x_ecdf, y_ecdf = get_empirical_cdf(excesses)
    ax2.plot(x_ecdf, y_ecdf, marker='.', linestyle='none', color='darkorange')
    ax2.set_title('The Empirical culmulative distribution function of Excesses (ECDF)', fontsize=16)
    ax2.set_xlabel('Excess over threshold')
    ax2.set_ylabel('Culmulative Probability')
    
    plt.tight_layout()
    plt.show()


from scipy.stats import genpareto

# ----- Fit the General Pareto Ditstibution ----- #
# Given the value to derive how long is the time that this event will happen again ?
shape, loc, scale = genpareto.fit(excesses, floc=0)
rate_of_exceedance = len(exceedances) / len(mock_data)

prob_excess_greater_than_max = genpareto.sf(threshold, c=shape, scale=scale) # survive function

unconditional_prob = rate_of_exceedance * prob_excess_greater_than_max # Unconditional probability
return_period = 1 / unconditional_prob

print(f"The unconditional probability of a value that over the threshold ({threshold:.4f}) is {unconditional_prob:.6f}")
print(f"Every {return_period:.0f} time units occurs once")

# ----- VaR ----- #
# In the contrrast, given the time period to compute the exceed value 
print("----- Value at Risk  -----")
T_return_period = 500 # 500 years one event
prob_T = 1 / T_return_period
conditional_prob_T = prob_T / rate_of_exceedance
return_level_excess = genpareto.ppf(1 - conditional_prob_T, c=shape, scale=scale)
return_level_actual = threshold + return_level_excess
print(f"An event with period {T_return_period}  years, The excess will be: {return_level_excess:.4f}")
print(f"that is, with threshold is {threshold:.4f}, the actual value shall be {return_level_actual:.4f}")

# ----- visualization of the ECDF fitness ----- #
plt.figure(figsize=(15, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Compute ECDF from data
x_ecdf, y_ecdf = get_empirical_cdf(excesses)
plt.plot(x_ecdf, y_ecdf, marker='.', linestyle='none', color='darkorange', label='ECDF')

# Plot the fitted CDF curve of the GPD model (theoritical model)
x_gpd = np.linspace(0, excesses.max() * 1.1, 200)
y_gpd = genpareto.cdf(x_gpd, c=shape, scale=scale)
plt.plot(x_gpd, y_gpd, 'b-', lw=2, label=f'fitted GPD model (ξ={shape:.2f}, σ={scale:.2f})')

plt.title('Excess ECDF vs. Fitted GPD Model', fontsize=16)
plt.xlabel('Excess over threshold')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.show()
