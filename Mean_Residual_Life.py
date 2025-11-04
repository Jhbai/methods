def mrl_threshold_selection(residuals):
    """
    使用Mean Residual Life Plot自動選擇閾值
    這個方法基於：當閾值合適時，MRL圖應該接近線性
    """
    abs_residuals = np.sort(np.abs(residuals))
    n = len(abs_residuals)
    
    # 候選閾值
    candidate_indices = np.arange(int(0.9 * n), int(0.99 * n))
    candidate_thresholds = abs_residuals[candidate_indices]
    
    best_threshold = None
    best_linearity = -np.inf
    
    for idx, u in zip(candidate_indices, candidate_thresholds):
        excesses = abs_residuals[idx:] - u
        
        if len(excesses) < 10:
            continue
        
        # 計算平均超過殘差
        mrl = np.mean(excesses)
        
        # 評估線性度（使用最小二乘法）
        x = np.arange(len(excesses))
        if len(x) > 1:
            _, _, r_value, _, _ = stats.linregress(x, excesses)
            linearity_score = r_value ** 2
            
            if linearity_score > best_linearity:
                best_linearity = linearity_score
                best_threshold = u
    
    return best_threshold
