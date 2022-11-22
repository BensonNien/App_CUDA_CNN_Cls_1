#pragma once
/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description: CUDA ver.
******************************************************************************/

#include <math.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <chrono>//random seed
#include <random> // normal_distribution random
#include <cmath>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "CUDACNNDataset.cuh"

// Utility
namespace CUDA_Algo_Lib
{
	float EvlElapsedTime();
	void RandomMatrix(size_t size_row, size_t size_col, float* p_kernel);
	void ConvNValid(float* p_matrix, float* p_kernel, size_t map_size_row, size_t map_size_col, size_t kernel_size_row, size_t kernel_size_col, float* p_outmatrix);// m n is the dimension of matrix and km kn is the dimension of kernel_, outmatrix is result
	void ActiveRelu(float* p_matrix, float bias, size_t m, size_t n);// m n is the dimension of matrix
	void CalExpone(float* p_matrix, float bias, size_t m, size_t n);// m n is the dimension of matrix
	void CalConvArrayPlus(float* p_x, float* p_y, size_t m, size_t n);
	void CalFCHArrayPlus(float* p_x, float* p_y, size_t m, size_t n);
	void CalSampArrayPlus(float* p_x, float* p_y, size_t m, size_t n);
	void CalArrayPlus(float* p_x, float* p_y, size_t m, size_t n);
	void ScaleMatrix(float* p_matrix, CUDA_Algo_Lib::RectSize scale, size_t matrix_rows, size_t matrix_cols, float* p_out_matrix);//sampling
	void Rot180(float* p_matrix, size_t m, size_t n, float* p_rot_matrix);
	void ConvNSampFull(float* p_matrix, float* p_kernel, size_t m, size_t n, size_t km, size_t kn, float* p_out_matrix, float* p_extend_matrix);// convn full mode
	void MatrixDrelu(float** p_matrix, size_t m, size_t n, float** M);// calculate derivation of ReLU function with matrix
	void MatrixDreluFChidden(float* p_matrix, size_t m, size_t n, float* p_M);// calculate derivation of ReLU function in FChiddenlayer with matrix
	void MatrixDreluConv(float* p_matrix, size_t m, size_t n, float* p_M);// calculate derivation of ReLU function in Convlayer with matrix
	void MatrixDsigmoid(float** p_matrix, size_t m, size_t n, float** p_M);// calculate derivation of sigmoid function with matrix
	void MatrixDsigmoidFChidden(float** p_matrix, size_t m, size_t n, float* p_M);// calculate derivation of sigmoid function in FChiddenlayer with matrix
	void Kronecker(float** p_matrix, CUDA_Algo_Lib::RectSize scale, size_t m, size_t n, float** p_outmatrix);
	void CalKronecker(float* p_nextlayer_matrix, CUDA_Algo_Lib::RectSize scale, size_t nextlayer_matrix_rows, size_t nextlayer_matrix_cols, float* p_out_matrix, size_t layer_out_matrix_rows, size_t layer_out_matrix_cols);
	void MatrixMultiply(float** p_matrix1, float** p_matrix2, size_t m, size_t n);//inner product of matrix 1 and matrix 2, result is matrix1
	void CalMatrixMultiply(float* p_matrix1, float* p_matrix2, size_t m, size_t n);
	void CalErrorsSum(float* p_errors, size_t idx_outmap, size_t outmap_num, size_t outmap_rows, size_t outmap_cols, size_t batch_size, float* p_m);
	float CalErrorSum(float* p_error, size_t m, size_t n);
	void CalArrayDivide(float* p_matrix, size_t batch_size, size_t m, size_t n);// result is matrix;
	void CalArrayMultiply(float* p_matrix, float val, size_t m, size_t n);// array multiply a float value, result in matrix
	void SetInLayerValue(float* p_maps, float** p_sum, size_t m, size_t n);
	void SetKernelValue(float* p_maps, float* p_sum, size_t m, size_t n);
	size_t FindIndex(float* p_batch_maps, size_t map_num, size_t map_rows, size_t map_cols);
	size_t FindIndex(float* p_batch_labels, size_t map_num);

	// CUDACNNLayer

	class CUDACNNLayer
	{
	private:
		size_t in_map_num_;
		size_t out_map_num_;
		char layer_type_;
		CUDA_Algo_Lib::RectSize map_size_;
		CUDA_Algo_Lib::RectSize scale_size_;
		CUDA_Algo_Lib::RectSize kernel_size_;
		size_t in_element_num_;
		size_t out_element_num_;
		size_t class_num_;

	public:
		CUDACNNLayer() = default;
		~CUDACNNLayer() {};

		std::vector<float> vec_kernel_;
		std::vector<float> vec_laststep_delta_kernel_;//for adding momentum
		std::vector<float> vec_output_maps_;
		std::vector<float> vec_errors_;
		std::vector<float> vec_bias_;

		CUDACNNLayer CreateInputLayer(size_t input_map_num, CUDA_Algo_Lib::RectSize map_size);
		CUDACNNLayer CreateConvLayer(size_t input_map_num, size_t output_map_num, CUDA_Algo_Lib::RectSize kernel_size);
		CUDACNNLayer CreateSampLayer(CUDA_Algo_Lib::RectSize scale_size);
		CUDACNNLayer CreateFullyConnectedHiddenLayer(size_t input_element_num, size_t output_element_num, size_t class_num);
		//CUDACNNLayer CreateOutputLayer(size_t input_element_num, size_t output_element_num, size_t class_num);
		CUDACNNLayer CreateOutputLayer(size_t class_num);

		void InitKernel(size_t front_map_num);
		void InitLastStepDeltaKernel(size_t front_map_num);//for adding momentum
		void InitOutputKernel(size_t front_map_num, CUDA_Algo_Lib::RectSize Kernel_size);
		void InitOutputLastStepDeltaKernel(size_t front_map_num, CUDA_Algo_Lib::RectSize Kernel_size);//for adding momentum
		void InitErros(size_t batch_size);
		void InitOutputMaps(size_t batch_size);
		void InitBias(size_t front_map_num, size_t idx_iter);

		void SetError(size_t num_batch, size_t map_no, size_t map_x, size_t map_y, float error_val);
		float* GetError(size_t num_batch, size_t map_no) {
			size_t shift_idx_error_batch_map = num_batch * out_map_num_ * map_size_.rows_ * map_size_.cols_;
			size_t shift_idx_error_out_map = map_no * map_size_.rows_ * map_size_.cols_;
			return (vec_errors_.data() + shift_idx_error_batch_map + shift_idx_error_out_map);
		}
		void SetFCHLayerError(size_t num_batch, size_t map_no, float* p_matrix, size_t m, size_t n);
		void SetSampLayerError(size_t num_batch, size_t map_no, float* p_matrix, size_t m, size_t n);
		void SetConvLayerError(size_t num_batch, size_t map_no, float* p_matrix, size_t m, size_t n);
		float* GetKernel(size_t num_batch, size_t map_no) {
			size_t shift_idx_front_map = num_batch * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_;
			size_t shift_idx_out_map = map_no * kernel_size_.rows_ * kernel_size_.cols_;
			return (vec_kernel_.data() + shift_idx_front_map + shift_idx_out_map);
		}
		size_t GetOutMapNum() {
			return out_map_num_;
		}
		char GetType() {
			return layer_type_;
		}
		CUDA_Algo_Lib::RectSize GetMapSize() {
			return map_size_;
		}
		void SetMapSize(CUDA_Algo_Lib::RectSize map_size) {
			this->map_size_ = map_size;
		}
		void SetOutMapNum(size_t out_map_num) {
			this->out_map_num_ = out_map_num;
		}
		CUDA_Algo_Lib::RectSize GetKernelSize() {
			return kernel_size_;
		}
		CUDA_Algo_Lib::RectSize GetScaleSize() {
			return scale_size_;
		}

	};

}