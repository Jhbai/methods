#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "pelt.h"

// Using algebra to reduce T(n) from O(n) to O(1)
static float cost(const float* cum_sum, const float* cum_sum2, size_t start, size_t end){
	size_t n = end - start;
	if(n <= 1)return 0.0f;
	
	// Compute negative log likelihood
	float sum = cum_sum[end] - cum_sum[start];
    float sum2 = cum_sum2[end] - cum_sum2[start];
    float mean = sum / n;

	float variance = (sum2/n) - (mean*mean); // E[X^2] - E[X]^2
	variance = variance*n/(n-1); // since it's sample variance
	if(variance <= 1e-9f)return 0.0f;
	
	// Equivalent Simplified Mathematical Formulation
	float n_float = (float)n;
	float nlog_likelihood = (n_float/2)*logf(2*M_PI*variance)+(n_float-1)/2;
	return nlog_likelihood;
}

void pelt(const float* data, size_t n_series, float penalty, int* changepoints, int* n_changepoints){
	// Prsum
	float* cum_sum = (float*)malloc((n_series + 1) * sizeof(float));
    float* cum_sum2 = (float*)malloc((n_series + 1) * sizeof(float));
	cum_sum[0] = 0.0f;
    cum_sum2[0] = 0.0f;
    for (size_t i = 0; i < n_series; i++) {
        cum_sum[i + 1] = cum_sum[i] + data[i];
        cum_sum2[i + 1] = cum_sum2[i] + data[i] * data[i];
    }

	float* F = (float*)malloc((n_series + 1)*sizeof(float)); // Cost Array
	int* cp = (int*)malloc((n_series + 1)*sizeof(int)); // ChangePoint Array
	int* R = (int*)malloc((n_series + 1)*sizeof(int)); // Candidate array
	float* costs_for_t = (float*)malloc((n_series + 1) * sizeof(float)); // Memory storage for cost, avoid repetition computation
	
	// Assign cost matrix
	F[0] = -penalty;
	for(size_t i = 1; i <= n_series; i++)
		F[i] = FLT_MAX;
	
	int R_len = 1;
	R[0] = 0;
	for(size_t t = 1; t <= n_series; t++){
		float min_val = FLT_MAX;
		int best_tau_idx = -1;
		
		// Find minimum cost in a loop
		for(size_t i = 0; i < R_len; i++){
			size_t tau = R[i]; // tau is the start point of the segments

			// O(1) Compute cost
			float cost_tau_t = cost(cum_sum, cum_sum2, tau, t);
			float cost_no_penalty = F[tau] + cost_tau_t;

			// Temporary storage for second loop
			costs_for_t[i] = cost_no_penalty;
			float curr_cost = cost_no_penalty + penalty; // means set tau as changpoint, how much cost it will cause

			if(curr_cost < min_val){
				min_val = curr_cost;
				best_tau_idx = tau;
			}
		}
		F[t] = min_val;
		cp[t] = best_tau_idx;

		size_t new_R_len = 0;
		for(size_t i = 0; i < R_len; i++){
			size_t tau = R[i];
			// Directly use the temp storage, avoid calling cost function
			if (costs_for_t[i] <= min_val)
				R[new_R_len++] = tau;
		}
		R[new_R_len++] = t; // Take t as a candidate
		R_len = new_R_len;
	}

	// Backtracing
	*n_changepoints = 0;
	size_t curr_cp = n_series;
	while(curr_cp > 0){
		(*n_changepoints)++;
		curr_cp = cp[curr_cp];
	}
	curr_cp = n_series;
	size_t i = *n_changepoints - 1;
	while(curr_cp > 0){
		changepoints[i--] = curr_cp;
		curr_cp = cp[curr_cp];
	}
	free(F);
	free(cp);
	free(R);
	free(cum_sum);
    free(cum_sum2);
    free(costs_for_t);
}
