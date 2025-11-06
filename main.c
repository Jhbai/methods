#include <stdio.h>
#include <stdlib.h> // For posix_memalign
#include <string.h> // For memcpy
#include <immintrin.h> // For AVX

// Dynamic malloc the 32 bits alignment memory address
float* align_malloc(size_t n){
	void* ptr = NULL;
	posix_memalign(&ptr, 32, sizeof(float)*n);
	return (float*)ptr;
}


void pointwise_slope(float* x_align, float* y_align, float* dst, size_t n){
	// Assign the malloc size
	size_t return_size = n*(n-1)/2;
	size_t align_size = ((n+7)/8)*8;
	
	// Slopes memory result
	size_t idx = 0;
	__m256 e_vec = _mm256_set1_ps(0.00001f);
	float* temp = align_malloc(8); // notice !!
	for(size_t i = 0; i < n; i++){
		__m256 x_i_vec = _mm256_set1_ps(x_align[i]);
		__m256 y_i_vec = _mm256_set1_ps(y_align[i]);
		
		// From i+1 to the start point of avx
		size_t j = i+1;
		size_t j_start_avx = ((i+1+7)/8)*8; // To the min distance of 8 multiplier from i+1
		for(;j < j_start_avx && j < n; j++){
			float delta_y = y_align[j] - y_align[i];
			float delta_x = x_align[j] - x_align[i];
			dst[idx++] = delta_y/(delta_x + 0.00001f);
		}
		for(j = j_start_avx; j < align_size; j+=8){
			__m256 x_j_vec = _mm256_load_ps(&x_align[j]);
			__m256 y_j_vec = _mm256_load_ps(&y_align[j]);
			__m256 delta_y_vec = _mm256_sub_ps(y_j_vec, y_i_vec);
			__m256 delta_x_vec = _mm256_sub_ps(x_j_vec, x_i_vec);
			delta_x_vec = _mm256_add_ps(delta_x_vec, e_vec);
			__m256 dst_vec = _mm256_div_ps(delta_y_vec, delta_x_vec);
			_mm256_store_ps(temp, dst_vec);
			for(int k = 0; k < 8; k++){
				if(j+k < n) dst[idx++] = temp[k];
			}
		}
	}
	free(temp);
}

int cmp_flt(const void* a, const void* b){
	float float_a = *(const float *)a;
	float float_b = *(const float *)b;

	if(float_a < float_b) return -1;
	else if(float_a > float_b) return 1;
	else return 0;
}

float median(float* arr, size_t n){
	if(n==0) return 0.0f;
	int mid = n/2;
	qsort(arr, n, sizeof(float), cmp_flt);
	if(n%2==0) return (arr[mid]+arr[mid-1])/2;
	else return arr[mid];

}

void theil_sen(float*x, float*y, float* res, size_t n){
	// Alloc the intercept pointer
	float* intercepts = (float*)malloc(n*sizeof(float));

	// Malloc aligned memory before intercept
	int align_size = ((n+7)/8)*8;
	float* x_align = align_malloc(align_size);
	float* y_align = align_malloc(align_size);
	memcpy(x_align, x, n*sizeof(float));
	memcpy(y_align, y, n*sizeof(float));
	for(size_t i = n; i < align_size; i++){
		x_align[i] = 0.0f;
		y_align[i] = 0.0f;
	}

  // Compute the slope
	float *slopes = (float*)malloc((n*(n-1)/2)*sizeof(float));
	pointwise_slope(x_align, y_align, slopes, n);
	float beta = median(slopes, n*(n-1)/2);
	
	// Compute the intercepts
	int idx = 0;
	float* tmp = align_malloc(8); // notice !!
	__m256 slope_vec = _mm256_set1_ps(beta);
	int n_vec = (n/8)*8;
	for(size_t i = 0; i < n_vec; i+=8){
		__m256 x_vec = _mm256_load_ps(&x_align[i]);
		__m256 y_vec = _mm256_load_ps(&y_align[i]);
		__m256 tmp_vec = _mm256_mul_ps(x_vec, slope_vec);
		tmp_vec = _mm256_sub_ps(y_vec, tmp_vec);
		_mm256_store_ps(tmp, tmp_vec);
		for(int j = 0; j<8; j++) intercepts[idx++] = tmp[j];
	}
	for(size_t i = n_vec; i < n; i++){
		float intercept = y_align[i] - x_align[i]*beta;
		intercepts[idx++] = intercept;
	}

	// Compute final intercept
	float alpha = median(intercepts, n);
	free(x_align);
	free(y_align);
	free(slopes);
	free(intercepts);
	free(tmp);
	res[0] = alpha;
	res[1] = beta;
}
 

int main(int argc, char** argv){
  
  size_t n = 9;
  float y[9] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};
  float x[9] = {3.2, 1.5, 0.8, 4.4, 3.3, 2.7, 6.6, 1.9, 2.4};
  float results[2];
  theil_sen(x, y, results, n);
  printf("\n(Model: y = %f + %f * x)\n", results[0], results[1]);
  return 0;
}

// gcc -o main main.c -mavx