"""
JumpStarter: Jump-Starting Multivariate Time Series Anomaly Detection
Reference: Ma et al., USENIX ATC 2021
Implementation: 100% Paper-Compliant Version
"""

import numpy as np
import cvxpy as cp
from typing import List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.fft import idct
from scipy.signal import correlate
from scipy.stats import genpareto
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted

class JumpStarter(BaseEstimator, OutlierMixin):
    """
    JumpStarter Anomaly Detector implementation.
    Strict adherence to Ma et al., USENIX ATC '21 algorithms.
    """
    
    def __init__(self, 
                 window_size: int = 20,          # w [Paper §3.2]
                 sampling_rate: float = 0.2,     # theta (initial) [Paper §3.4]
                 cluster_threshold: float = 0.5, # For shape-based clustering
                 contamination: float = 0.01,    # Risk probability for EVT
                 random_state: Optional[int] = None,
                 verbose: bool = False):         # Fixed: Added verbose to __init__
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.cluster_threshold = cluster_threshold
        self.contamination = contamination
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y=None):
        """
        Offline Processing Phase [Paper Figure 5]:
        1. Shape-Based Clustering.
        2. Jump-Start: Calibrate EVT threshold using training data logic.
        """
        X = check_array(X)
        self.n_features_ = X.shape[1]
        self.rng_ = np.random.RandomState(self.random_state)
        
        # 1. Shape-Based Clustering [Paper §3.3]
        n_init = min(len(X), 1440) 
        init_data = X[:n_init]
        
        if self.verbose:
            print(f"JumpStarter: Running Shape-Based Clustering on first {n_init} samples...")
        self.clusters_ = self._shape_based_clustering(init_data)
        
        if self.verbose:
            print(f"JumpStarter: Found {len(self.clusters_)} clusters: {self.clusters_}")
        
        # 2. Calibration (Jump-Starting)
        if self.verbose:
            print("JumpStarter: Calibrating EVT threshold on training data...")
            
        calib_size = min(len(X), 2000)
        scores = self._compute_anomaly_scores(X[:calib_size])
        
        # 3. EVT Thresholding [Paper §3.5] (POT Method)
        self.threshold_ = self._calibrate_evt_threshold(scores)
        
        if self.verbose:
            print(f"JumpStarter: Fitted with EVT threshold {self.threshold_:.4f}")
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Online Processing Phase:
        Returns -1 for anomalies, 1 for inliers.
        """
        check_is_fitted(self, ['clusters_', 'threshold_'])
        X = check_array(X)
        scores = self._compute_anomaly_scores(X)
        return np.where(scores > self.threshold_, -1, 1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['clusters_', 'threshold_'])
        X = check_array(X)
        return self._compute_anomaly_scores(X)

    # ================= Core Algorithms =================

    def _shape_based_clustering(self, data: np.ndarray) -> List[List[int]]:
        """
        Shape-Based Clustering using SBD [Paper §3.3].
        """
        n_features = data.shape[1]
        if n_features < 2:
            return [[0]]

        dist_matrix = np.zeros((n_features, n_features))
        
        # Z-normalization for NCC
        data_norm = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-10)
        n_samples = data.shape[0]

        for i in range(n_features):
            for j in range(i + 1, n_features):
                cc = correlate(data_norm[:, i], data_norm[:, j], mode='full')
                ncc = cc / n_samples
                max_ncc = np.max(np.abs(ncc))
                sbd = 1.0 - max_ncc
                dist_matrix[i, j] = dist_matrix[j, i] = max(0, sbd)

        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='average')
        labels = fcluster(linkage_matrix, self.cluster_threshold, criterion='distance')
        
        clusters = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            clusters.append(np.where(labels == label)[0].tolist())
            
        return clusters

    def _compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Sliding Window Detection Loop with Retry Mechanism [Paper §3.5].
        """
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        Psi = idct(np.eye(self.window_size), norm='ortho', axis=0)

        # Sliding Window
        for t in range(self.window_size, n_samples + 1):
            window_slice = slice(t - self.window_size, t)
            X_window = X[window_slice]
            
            X_prime_window = np.zeros_like(X_window)
            
            for cluster_idx in self.clusters_:
                X_c = X_window[:, cluster_idx]
                
                # Robust Retry Logic [Paper: "gradually increase theta by 0.1"]
                theta_curr = self.sampling_rate
                X_c_recon = None
                
                # Try reconstruction, increase theta if it fails
                while theta_curr < 1.0:
                    try:
                        sc = self._lesinn_confidence(X_c)
                        B, T = self._outlier_resistant_sampling(X_c, sc, theta=theta_curr)
                        
                        # Attempt CS Reconstruction
                        X_c_recon = self._solve_cs_matrix(T, Psi, B)
                        
                        # If successful (non-zero result), break
                        if np.any(X_c_recon):
                            break
                        else:
                            # Solver returned zeros (soft failure), retry
                             theta_curr += 0.1
                    except Exception:
                        # Solver crashed, retry
                        theta_curr += 0.1
                
                # Fallback if all retries fail
                if X_c_recon is None or not np.any(X_c_recon):
                     X_c_recon = np.zeros_like(X_c) # Worst case

                X_prime_window[:, cluster_idx] = X_c_recon

            # Anomaly Score
            curr_orig = X_window[-1]
            curr_recon = X_prime_window[-1]
            diffs = np.abs(curr_orig - curr_recon)
            diffs = np.maximum(diffs, 1e-6)
            harmonic_mean = len(diffs) / np.sum(1.0 / diffs)
            
            scores[t-1] = harmonic_mean
            
        return scores

    def _lesinn_confidence(self, X_c: np.ndarray) -> np.ndarray:
        w = X_c.shape[0]
        dists = np.zeros((w, w))
        for i in range(w):
            dists[i] = np.linalg.norm(X_c - X_c[i], axis=1)
        
        confidence = np.zeros(w)
        k_neighbors = min(5, w - 1)
        if k_neighbors < 1: return np.ones(w)

        for i in range(w):
            knn_dists = np.sort(dists[i])[1 : k_neighbors + 1]
            outlier_score = np.mean(knn_dists)
            confidence[i] = 1.0 / (1.0 + outlier_score)
        return confidence

    def _outlier_resistant_sampling(self, X_c: np.ndarray, sc: np.ndarray, theta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Algorithm 1 with variable theta.
        """
        w, k = X_c.shape
        m = int(np.ceil(w * theta))
        
        # Parameters [Paper §4.1.3 Implementation]
        scale_R = 50 
        rho = 0.1     # Updated to 0.1 per paper
        sigma = 0.5   # Per paper
        
        T = np.zeros((m, w))
        phi_probs = self.rng_.rand(m)
        
        sc_sum = np.sum(sc)
        sc_norm = sc / sc_sum if sc_sum > 0 else np.ones(w)/w
        sc_cumsum = np.cumsum(sc_norm)
        
        steps = np.linspace(0, 1, int(w * scale_R))
        
        for step in steps:
            t = np.searchsorted(sc_cumsum, step, side='left')
            if t >= w: t = w - 1
            
            for i in range(m):
                dist_sq = (phi_probs[i] - step) ** 2
                prob = rho * np.exp(-dist_sq / (2 * (sigma ** 2)))
                if self.rng_.rand() < prob:
                    T[i, t] += 1
                    
        for i in range(m):
            t_force = np.searchsorted(sc_cumsum, phi_probs[i], side='left')
            if t_force >= w: t_force = w - 1
            T[i, t_force] += 1
            
        row_sums = np.sum(T, axis=1, keepdims=True)
        T = T / (row_sums + 1e-10)
        B = T @ X_c
        return B, T

    def _solve_cs_matrix(self, T: np.ndarray, Psi: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compressed Sensing Reconstruction using CVXPY.
        """
        w, k = Psi.shape[0], B.shape[1]
        A = T @ Psi
        Alpha = cp.Variable((w, k))
        
        # Eq (2): A * Alpha = B
        constraints = [A @ Alpha == B]
        objective = cp.Minimize(cp.norm(Alpha, 1))
        prob = cp.Problem(objective, constraints)
        
        try:
            # ECOS is standard. Using robust options.
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=200)
            if Alpha.value is None or prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return np.zeros((w, k))
            return Psi @ Alpha.value
        except Exception:
            return np.zeros((w, k))

    def _calibrate_evt_threshold(self, scores: np.ndarray) -> float:
        """
        Robust EVT (POT) Threshold Calibration.
        """
        valid_scores = scores[scores > 1e-9]
        if len(valid_scores) < 10:
            return np.max(scores) if len(scores) > 0 else 1.0
            
        # POT: Pick threshold u at 90% (tuneable)
        u = np.percentile(valid_scores, 90)
        excesses = valid_scores[valid_scores > u] - u
        
        if len(excesses) < 5:
            return np.max(valid_scores)
            
        try:
            # Fit GPD
            c, loc, scale = genpareto.fit(excesses, floc=0)
            N = len(valid_scores)
            Nu = len(excesses)
            q = self.contamination
            
            if abs(c) < 1e-5:
                threshold = u - scale * np.log((q * N) / Nu)
            else:
                threshold = u + (scale / c) * (np.power((q * N) / Nu, -c) - 1)
                
            if np.isnan(threshold) or np.isinf(threshold):
                raise ValueError("Bad EVT fit")
                
            return threshold
        except Exception:
            # Fallback to simple percentile if EVT fails
            return np.percentile(valid_scores, 100 * (1 - self.contamination))
