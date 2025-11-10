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

if __name__ == "__main__":
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
