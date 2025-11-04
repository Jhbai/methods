import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def pot_threshold_selection(residuals, initial_percentile=90):
    """
    使用POT方法選擇閾值
    Parameters:
    -----------
    residuals: (np.ndarray)Theil-Sen估計器計算出的殘差絕對值
    initial_percentile: (float)初始閾值的百分位數（建議90-95）
    Returns:
    -----------
    optimal_threshold: (float)選擇的閾值
    xi: (float)GPD形狀參數
    sigma: (float)GPD尺度參數
    -----------
    """
    abs_residuals = np.abs(residuals)
    
    # 候選閾值：從90百分位到99百分位
    percentiles = np.linspace(initial_percentile, 99, 20)
    candidate_thresholds = np.percentile(abs_residuals, percentiles)
    
    best_threshold = None
    best_score = -np.inf
    best_params = None
    
    for u in candidate_thresholds:
        # 提取超過閾值的殘差
        excesses = abs_residuals[abs_residuals > u] - u
        
        if len(excesses) < 10:  # 至少需要10個樣本
            continue
            
        try:
            # 擬合GPD分布
            params = stats.genpareto.fit(excesses, floc=0)
            xi_hat, loc, sigma_hat = params
            
            # 使用Anderson-Darling檢驗評估擬合優度
            ad_stat = stats.anderson(excesses, dist='genpareto').statistic
            
            # 計算分數（負的AD統計量，越大越好）
            score = -ad_stat
            
            if score > best_score:
                best_score = score
                best_threshold = u
                best_params = (xi_hat, sigma_hat)
                
        except:
            continue
    
    return best_threshold, best_params[0], best_params[1]


def calculate_anomaly_threshold(residuals, exceedance_probability=0.01):
    """
    基於POT方法計算異常檢測閾值
    
    Parameters:
    -----------
    residuals: (np.ndarray)殘差
    exceedance_probability: (float)期望的超過概率（如0.01表示1%的假陽性率）
    Returns:
    -----------
    anomaly_threshold: (float)異常檢測閾值
    -----------
    """
    # 步驟1：選擇POT閾值
    pot_threshold, xi, sigma = pot_threshold_selection(residuals)
    
    abs_residuals = np.abs(residuals)
    n_total = len(abs_residuals)
    n_exceedances = np.sum(abs_residuals > pot_threshold)
    
    # 步驟2：估計超過概率
    # P(X > x) = (n_u / n) * P(X - u > x - u | X > u)
    # 其中 P(X - u > x - u | X > u) = (1 + xi * (x-u) / sigma)^(-1/xi)
    
    zeta = n_exceedances / n_total  # 超過初始閾值的比例
    
    # 計算對應於目標超過概率的閾值
    if xi != 0:
        anomaly_threshold = pot_threshold + (sigma / xi) * \
                           ((exceedance_probability / zeta) ** (-xi) - 1)
    else:
        # 當xi=0時，GPD退化為指數分布
        anomaly_threshold = pot_threshold - sigma * np.log(exceedance_probability / zeta)
    
    return anomaly_threshold, pot_threshold, xi, sigma


# 使用範例
def example_usage():
    # 假設你已經計算出殘差
    # residuals = actual_values - theil_sen_predictions
    
    # 生成示例數據（實際使用時替換為你的殘差）
    np.random.seed(42)
    normal_residuals = np.random.normal(0, 1, 10000)
    extreme_residuals = np.random.exponential(5, 100)
    residuals = np.concatenate([normal_residuals, extreme_residuals])
    
    # 計算異常閾值（1%假陽性率）
    threshold, pot_u, xi, sigma = calculate_anomaly_threshold(
        residuals, exceedance_probability=0.01
    )
    
    print(f"POT初始閾值: {pot_u:.4f}")
    print(f"GPD形狀參數 ξ: {xi:.4f}")
    print(f"GPD尺度參數 σ: {sigma:.4f}")
    print(f"異常檢測閾值: {threshold:.4f}")
    print(f"預期假陽性率: 1%")
    
    # 檢測異常
    abs_residuals = np.abs(residuals)
    anomalies = abs_residuals > threshold
    print(f"檢測到的異常比例: {anomalies.sum() / len(residuals) * 100:.2f}%")
    return threshold

# 執行範例
if __name__ == "__main__":
    threshold = example_usage()
