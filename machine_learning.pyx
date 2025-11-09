# machine_learning.pyx (Final Corrected Version)

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# --- Import C header files ---
cdef extern from "dtw.h":
    ctypedef struct DTWConfig:
        float sakoe_chiba_band
        float threshold

    ctypedef struct DTWResult:
        int start
        int end
        float distance
        int path_length
        int* path_i
        int* path_j

    DTWResult* subsequence_alignment(const float* src, size_t src_len, const float* trg, size_t trg_len, const DTWConfig* config)
    void dtw_free_result(DTWResult* result)

cdef extern from "mann_kendall.h":
    void mann_kendall(const float* data, size_t n, long long* S, float* var_S, float* Z)

cdef extern from "pelt.h":
    void pelt(const float* data, size_t n_series, float penalty, int* changepoints, int* n_changepoints)

cdef extern from "theil_sen.h":
    void theil_sen(float* x, float* y, float* res, size_t n)
    void theil_sen_window(float*y, float* res, size_t n, size_t win_size)


# --- Python Wrapper Functions ---
# Initialize NumPy support
np.import_array()

def dtw_subsequence_alignment(np.ndarray[np.float32_t, ndim=1] src, 
                               np.ndarray[np.float32_t, ndim=1] trg, 
                               float sakoe_chiba_band=0.1, 
                               float threshold=0.0):
    """
    Python wrapper for the subsequence_alignment DTW function.
    Returns a dictionary with the results or None if no path is found.
    """
    cdef DTWConfig config
    cdef DTWResult* result
    cdef int* c_path_i
    cdef int* c_path_j
    cdef int i

    config.sakoe_chiba_band = sakoe_chiba_band
    config.threshold = threshold

    result = subsequence_alignment(&src[0], src.shape[0], &trg[0], trg.shape[0], &config)
    
    if result == NULL:
        return None

    try:
        path_i = np.empty(result.path_length, dtype=np.int32)
        path_j = np.empty(result.path_length, dtype=np.int32)
        
        c_path_i = result.path_i
        c_path_j = result.path_j
        
        for i in range(result.path_length):
            path_i[i] = c_path_i[i]
            path_j[i] = c_path_j[i]

        py_result = {
            'start': result.start,
            'end': result.end,
            'distance': result.distance,
            'path_length': result.path_length,
            'path_i': path_i,
            'path_j': path_j
        }
        return py_result
    finally:
        dtw_free_result(result)

def mk_test(np.ndarray[np.float32_t, ndim=1] data):
    """
    Python wrapper for the Mann-Kendall test.
    Returns a tuple of (S, var_S, Z).
    """
    cdef long long S = 0
    cdef float var_S = 0.0
    cdef float Z = 0.0
    
    mann_kendall(&data[0], data.shape[0], &S, &var_S, &Z)
    
    return S, var_S, Z

def pelt_change_point(np.ndarray[np.float32_t, ndim=1] data, float penalty):
    """
    Python wrapper for the PELT change point detection algorithm.
    Returns a NumPy array of change point indices.
    """
    # CORRECTED: 'cdef int i' is now moved to the top of the function.
    cdef int n_changepoints = 0
    cdef int* changepoints = <int*>malloc(data.shape[0] * sizeof(int))
    cdef int i

    if not changepoints:
        raise MemoryError()

    try:
        pelt(&data[0], data.shape[0], penalty, changepoints, &n_changepoints)
        
        # Copy results to a NumPy array of the correct size
        result_np = np.empty(n_changepoints, dtype=np.int32)
        for i in range(n_changepoints):
            result_np[i] = changepoints[i]
            
        return result_np
    finally:
        free(changepoints)

def ts_estimator(np.ndarray[np.float32_t, ndim=1] x, np.ndarray[np.float32_t, ndim=1] y):
    """
    Python wrapper for the Theil-Sen estimator.
    Returns a tuple of (intercept, slope).
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays x and y must have the same length.")
        
    cdef float res[2]
    theil_sen(&x[0], &y[0], res, x.shape[0])
    
    return res[0], res[1] # (alpha, beta)

def ts_window_estimator(np.ndarray[np.float32_t, ndim=1] y, int win_size):
    """
    Python wrapper for the sliding window Theil-Sen estimator.
    Returns a NumPy array with the regression baseline.
    """
    cdef np.ndarray[np.float32_t, ndim=1] res = np.empty_like(y)
    
    theil_sen_window(&y[0], &res[0], y.shape[0], <size_t>win_size)
    
    return res