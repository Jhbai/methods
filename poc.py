def generalized_esd_test(data, k, alpha=0.05):
    """
    Using Generalized ESD Test to detect at most K anomalies
    H0: there is no anomaly
    H1: at least one anomaly
    ---
    parameters:
    data (np.ndarray): the raw data of time seires
    k (int): the at most amount of anomalies
    alpha (float): the significance value
    
    ---
    return:
    indices: outlier indices of data
    values: outlier values
    """
    x = list(data)
    n = len(x)
    indices = []
    
    for i in range(k):
        current_n = n - i
        # ----- If data size is too small, then no data shall be return ----- #
        if current_n < 3:
            break

        # ----- Compute the statistics (R) ----- #
        mean_x = np.mean(x)
        std_x = np.std(x, ddof=1)
        residuals = np.abs(x - mean_x)
        max_residual_idx = np.argmax(residuals)
        r_calculated = np.max(residuals) / std_x

        # ----- Compute the critical value (lambda) ----- #
        p = 1 - alpha / (2 * current_n)
        t_crit = stats.t.ppf(p, current_n - 2)
        lambda_critical = (current_n - 1) * t_crit / np.sqrt((current_n - 2 + t_crit**2) * current_n)

        if r_calculated > lambda_critical:
            original_idx = np.where(data == x[max_residual_idx])[0][0]
            outlier_indices.append(original_idx)
            x.pop(max_residual_idx)

    values = np.array(data)[outlier_indices]
    return indices, values

  
