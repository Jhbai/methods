def lmoments_threshold_selection(residuals):
    """
    使用L-moments方法自動選擇閾值
    
    基於：GPD有特定的L-skewness和L-kurtosis關係
    τ4 = τ3 * (1 + 5*τ3) / (5 + τ3)
    """
    abs_residuals = np.sort(np.abs(residuals))
    n = len(abs_residuals)
    
    # 候選閾值（從25%到95%百分位）
    percentiles = np.linspace(25, 95, 20)
    candidates = np.percentile(abs_residuals, percentiles)
    
    best_threshold = None
    min_distance = np.inf
    
    for u in candidates:
        excesses = abs_residuals[abs_residuals > u] - u
        
        if len(excesses) < 20:
            continue
        
        # 計算L-moments
        sorted_exc = np.sort(excesses)
        n_exc = len(sorted_exc)
        
        # L-moments估計
        b0 = np.mean(sorted_exc)
        b1 = np.mean([(i * sorted_exc[i]) for i in range(n_exc)]) / n_exc
        b2 = np.mean([(i * (i-1) * sorted_exc[i]) for i in range(2, n_exc)]) / (n_exc * (n_exc - 1))
        
        l1 = b0
        l2 = 2 * b1 - b0
        l3 = 6 * b2 - 6 * b1 + b0
        
        if l2 == 0:
            continue
            
        # L-moment比率
        t3 = l3 / l2  # L-skewness
        
        # GPD的理論關係
        def gpd_l4_from_l3(t3):
            return t3 * (1 + 5 * t3) / (5 + t3)
        
        # 計算實際的L-kurtosis（需要第4個L-moment）
        # 簡化版：直接使用與理論曲線的距離
        # 這裡我們使用t3與合理範圍的距離
        
        # GPD的t3範圍通常在(-1/3, 1)
        if -1/3 <= t3 <= 1:
            distance = abs(t3 - 0.2)  # 0.2是典型的重尾值
        else:
            distance = np.inf
        
        if distance < min_distance:
            min_distance = distance
            best_threshold = u
    
    return best_threshold
