/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description: GPU (CUDA) ver.
******************************************************************************/

#include "GPUCNNLayer.cuh"

// Utility
void GPUAlgoOp_Lib::RandomMatrix(size_t size_row, size_t size_col, float* p_kernel)
{
	// construct a trivial random generator engine from a time-based seed:
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<float> distribution(0, 0.14);

	std::cout << "--------------- kernel's content -----------------" << std::endl;
	for (size_t i = 0; i < size_row; i++)
	{
		for (size_t j = 0; j < size_col; j++)
		{
			p_kernel[i * size_col + j] = distribution(generator);
			std::cout << std::to_string(p_kernel[i * size_col + j]) << " ";
		}
		std::cout << std::endl;
	}
}

void GPUAlgoOp_Lib::ConvNValid(float* p_matrix, float* p_kernel, size_t map_size_row, size_t map_size_col, size_t kernel_size_row, size_t kernel_size_col, float* outmatrix)
{

	// the number of row of convolution
	size_t num_conv_row = map_size_row - kernel_size_row + 1;
	// the number of column of convolution
	size_t num_conv_col = map_size_col - kernel_size_col + 1;

	for (size_t i = 0; i < num_conv_row; i++)
	{
		for (size_t j = 0; j < num_conv_col; j++)
		{
			float sum = 0.0;
			for (size_t ki = 0; ki < kernel_size_row; ki++)
			{
				for (size_t kj = 0; kj < kernel_size_col; kj++)
				{
					sum += p_matrix[((i + ki) * map_size_col) + (j + kj)] * p_kernel[(ki * kernel_size_col) + kj];
				}
			}
			outmatrix[(i * num_conv_col) + j] = sum;
		}
	}
}

void GPUAlgoOp_Lib::ActiveRelu(float* p_matrix, float bias, size_t m, size_t n)
{

	float x1 = 0.0;
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			x1 = p_matrix[(i * n) + j] + bias;

			if (x1 > 0.0) {
				p_matrix[(i * n) + j] = x1;
			}
			else if (0.0 == x1 || x1 < 0.0) {

				p_matrix[(i * n) + j] = 0.0;

			}
			else {
				exit(0);
			}
		}
	}
}

void GPUAlgoOp_Lib::CalExpone(float* p_matrix, float bias, size_t m, size_t n)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			//std::cout << "Outputlayer's actual ouput(p_matrix[" << i << "][" << j << "] + bias_): " << p_matrix[i][j] + bias_ << std::endl;
			p_matrix[(i * n) + j] = exp(p_matrix[(i * n) + j] + bias);
			//std::cout << "Outputlayer's expone actual ouput: " << p_matrix[i][j] << std::endl;
		}
	}
}

void GPUAlgoOp_Lib::CalConvArrayPlus(float* p_x, float* p_y, size_t m, size_t n)
{

	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_y[(i * n) + j] += p_x[(i * n) + j];
		}
	}
}

void GPUAlgoOp_Lib::CalFCHArrayPlus(float* p_x, float* p_y, size_t m, size_t n)
{

	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_y[(i * n) + j] += p_x[(i * n) + j];
		}
	}
}

void GPUAlgoOp_Lib::CalSampArrayPlus(float* p_x, float* p_y, size_t m, size_t n)
{

	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_y[(i * n) + j] += p_x[(i * n) + j];
		}
	}
}

void GPUAlgoOp_Lib::CalArrayPlus(float* p_x, float* p_y, size_t m, size_t n)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_y[i * n + j] += p_x[i * n + j];
		}
	}
}

void GPUAlgoOp_Lib::ScaleMatrix(float* p_matrix, GPUCNNDataStruct_Lib::RectSizePKG scale, size_t matrix_rows, size_t matrix_cols, float* p_out_matrix)
{
	size_t out_matrix_rows = matrix_rows / scale.rows_;
	size_t out_matrix_cols = matrix_cols / scale.cols_;
	if (out_matrix_rows * scale.rows_ != matrix_rows || out_matrix_cols * scale.cols_ != matrix_cols)
	{
		std::cout << "scale can not divide by p_matrix";
	}

	float whole_s = (float)(scale.rows_ * scale.cols_);
	float sum = 0.0;
	for (size_t i = 0; i < out_matrix_rows; i++) {
		for (size_t j = 0; j < out_matrix_cols; j++) {
			sum = 0.0;
			for (size_t si = i * scale.rows_; si < (i + 1) * scale.rows_; si++) {
				for (size_t sj = j * scale.cols_; sj < (j + 1) * scale.cols_; sj++) {
					sum += p_matrix[(si * matrix_cols) + sj];
				}
			}
			p_out_matrix[(i * out_matrix_cols) + j] = sum / whole_s;
		}
	}
}

void GPUAlgoOp_Lib::Rot180(float* p_matrix, size_t m, size_t n, float* p_rot_matrix)
{

	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_rot_matrix[i * n + j] = p_matrix[i * n + (n - 1 - j)];
		}
	}

	for (size_t j = 0; j < n; j++)
	{
		for (size_t i = 0; i < m / 2; i++)
		{
			std::swap(p_rot_matrix[i * n + j], p_rot_matrix[(m - 1 - i) * n + j]);
		}
	}
}

void GPUAlgoOp_Lib::ConvNSampFull(float* p_matrix, float* p_kernel, size_t m, size_t n, size_t km, size_t kn, float* p_out_matrix, float* p_extend_matrix)
{

	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			p_extend_matrix[((i + km - 1) * n) + (j + kn - 1)] = p_matrix[i * n + j];
		}
	}

	ConvNValid(p_extend_matrix, p_kernel, (m + 2 * (km - 1)), (n + 2 * (kn - 1)), km, kn, p_out_matrix);

}

void GPUAlgoOp_Lib::MatrixDrelu(float** p_matrix, size_t m, size_t n, float** p_M)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{

			if (p_matrix[i][j] > 0.0)
			{
				p_M[i][j] = 1.0;
			}
			else if (0.0 == p_matrix[i][j] || p_matrix[i][j] < 0.0) {
				p_M[i][j] = 0.0;
			}
		}
	}
}

//for derivation of ReLU active fun. 
void GPUAlgoOp_Lib::MatrixDreluFChidden(float* p_matrix, size_t m, size_t n, float* p_M)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{

			if (p_matrix[i * n + j] > 0.0)
			{
				*p_M = 1.0;
			}
			else if (0.0 == p_matrix[i * n + j] || p_matrix[i * n + j] < 0.0) {
				*p_M = 0.0;
			}

		}
	}
}

void GPUAlgoOp_Lib::MatrixDreluConv(float* p_matrix, size_t m, size_t n, float* p_M)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{

			if (p_matrix[i * n + j] > 0.0)
			{
				*p_M = 1.0;
			}
			else if (0.0 == p_matrix[i * n + j] || p_matrix[i * n + j] < 0.0) {
				*p_M = 0.0;
			}

		}
	}
}

void GPUAlgoOp_Lib::MatrixDsigmoid(float** p_matrix, size_t m, size_t n, float** p_M)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_M[i][j] = p_matrix[i][j] * (1 - p_matrix[i][j]);
		}
	}
}

//for derivation of sigmoid active fun.
void GPUAlgoOp_Lib::MatrixDsigmoidFChidden(float** p_matrix, size_t m, size_t n, float* p_M)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			*p_M = p_matrix[i][j] * (1 - p_matrix[i][j]);
		}
	}
}

void GPUAlgoOp_Lib::Kronecker(float** p_matrix, GPUCNNDataStruct_Lib::RectSizePKG scale, size_t m, size_t n, float** p_OutMatrix)
{
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			for (size_t ki = i * scale.rows_; ki < (i + 1) * scale.rows_; ki++) {
				for (size_t kj = j * scale.cols_; kj < (j + 1) * scale.cols_; kj++) {
					p_OutMatrix[ki][kj] = p_matrix[i][j];
				}
			}
		}
	}
}

void GPUAlgoOp_Lib::CalKronecker(float* p_nextlayer_matrix, GPUCNNDataStruct_Lib::RectSizePKG scale, size_t nextlayer_matrix_rows, size_t nextlayer_matrix_cols, float* p_out_matrix, size_t layer_out_matrix_rows, size_t layer_out_matrix_cols)
{
	for (size_t i = 0; i < nextlayer_matrix_rows; i++) {
		for (size_t j = 0; j < nextlayer_matrix_cols; j++) {
			for (size_t ki = (i * scale.rows_); ki < ((i + 1) * scale.rows_); ki++) {
				for (size_t kj = (j * scale.cols_); kj < ((j + 1) * scale.cols_); kj++) {
					p_out_matrix[ki * layer_out_matrix_cols + kj] = p_nextlayer_matrix[i * nextlayer_matrix_cols + j];
				}
			}
		}
	}
}

void GPUAlgoOp_Lib::MatrixMultiply(float** p_matrix1, float** p_matrix2, size_t m, size_t n)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_matrix1[i][j] = p_matrix1[i][j] * p_matrix2[i][j];
		}
	}
}

void GPUAlgoOp_Lib::CalMatrixMultiply(float* p_matrix1, float* p_matrix2, size_t m, size_t n)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_matrix1[i * n + j] = p_matrix1[i * n + j] * p_matrix2[i * n + j];
		}
	}
}

void GPUAlgoOp_Lib::CalErrorsSum(float* p_errors, size_t idx_outmap, size_t outmap_num, size_t outmap_rows, size_t outmap_cols, size_t batch_size, float* p_m)
{
	float sum = 0.0;
	size_t shift_idx_error_batch_map = 0;
	size_t shift_idx_error_out_map = 0;
	size_t idx_error_out_map = 0;
	for (size_t mi = 0; mi < outmap_rows; mi++) {
		for (size_t nj = 0; nj < outmap_cols; nj++) {
			sum = 0.0;
			for (size_t i = 0; i < batch_size; i++) {
				shift_idx_error_batch_map = i * outmap_num * outmap_rows * outmap_cols;
				shift_idx_error_out_map = idx_outmap * outmap_rows * outmap_cols;
				idx_error_out_map = shift_idx_error_batch_map + shift_idx_error_out_map + (mi * outmap_cols) + nj;

				sum += p_errors[idx_error_out_map];
			}
			p_m[mi * outmap_cols + nj] = sum;
		}
	}
}

float GPUAlgoOp_Lib::CalErrorSum(float* p_error, size_t m, size_t n)
{
	float sum = 0.0;
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			sum += p_error[i * n + j];
		}
	}
	return sum;
}

void GPUAlgoOp_Lib::CalArrayDivide(float* p_M, size_t batchSize, size_t m, size_t n)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_M[i * n + j] = p_M[i * n + j] / batchSize;
		}
	}
}

void GPUAlgoOp_Lib::CalArrayMultiply(float* p_matrix, float val, size_t m, size_t n)
{

	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_matrix[i * n + j] *= val;
		}
	}
}

size_t GPUAlgoOp_Lib::FindIndex(float* p_batch_maps, size_t map_num, size_t map_rows, size_t map_cols)
{

	FILE* fy;
	fy = fopen("./outputdata/outputmaps_.txt", "a");
	/*
	if( (err=fopen_s(&fy, "outputmaps_.txt", "a")) != 0 )
		exit(1) ;
	*/
	size_t shift_idx_layer_out_map = 0 * map_rows * map_cols;
	size_t idx_layer_out_map = shift_idx_layer_out_map + (0 * map_cols + 0);
	size_t index = 0;
	float v;
	float Max = p_batch_maps[idx_layer_out_map];
	fprintf(fy, "%f ", Max);
	for (size_t i = 1; i < map_num; i++)
	{
		shift_idx_layer_out_map = i * map_rows * map_cols;
		idx_layer_out_map = shift_idx_layer_out_map + (0 * map_cols + 0);
		v = p_batch_maps[idx_layer_out_map];
		fprintf(fy, "%f\n", v);
		if (p_batch_maps[idx_layer_out_map] > Max)
		{
			Max = p_batch_maps[idx_layer_out_map];
			index = i;
		}
	}
	fclose(fy);
	return index;
}

size_t GPUAlgoOp_Lib::FindIndex(float* p_batch_labels, size_t map_num)
{
	size_t index = 0;
	float Max = p_batch_labels[0];
	for (size_t i = 1; i < map_num; i++)
	{
		float v = p_batch_labels[i];
		if (p_batch_labels[i] > Max)
		{
			Max = p_batch_labels[i];
			index = i;
		}
	}
	return index;
}

void GPUAlgoOp_Lib::SetInLayerValue(float* p_maps, float** p_sum, size_t m, size_t n)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_maps[i * n + j] = p_sum[i][j];
		}
	}
}

void GPUAlgoOp_Lib::SetKernelValue(float* p_maps, float* p_sum, size_t m, size_t n)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			p_maps[i * n + j] = p_sum[i * n + j];
		}
	}
}

// GPUCNNLayer
GPUCNNLayer GPUCNNLayer::CreateInputLayer(size_t input_map_num, GPUCNNDataStruct_Lib::RectSizePKG map_size)
{
	GPUCNNLayer layer;
	layer.layer_type_ = 'I';
	layer.in_map_num_ = input_map_num;
	layer.out_map_num_ = input_map_num;
	layer.map_size_ = map_size;
	return layer;
}
GPUCNNLayer GPUCNNLayer::CreateConvLayer(size_t input_map_num, size_t output_map_num, GPUCNNDataStruct_Lib::RectSizePKG kernel_size)
{
	GPUCNNLayer layer;
	layer.layer_type_ = 'C';
	layer.in_map_num_ = input_map_num;
	layer.out_map_num_ = output_map_num;
	layer.kernel_size_ = kernel_size;
	return layer;
}
GPUCNNLayer GPUCNNLayer::CreateSampLayer(GPUCNNDataStruct_Lib::RectSizePKG scale_size)
{
	GPUCNNLayer layer;
	layer.layer_type_ = 'S';
	layer.scale_size_ = scale_size;
	return layer;
}
GPUCNNLayer GPUCNNLayer::CreateFullyConnectedHiddenLayer(size_t input_element_num, size_t output_element_num, size_t class_num)
{
	GPUCNNLayer layer;
	layer.in_element_num_ = input_element_num;
	layer.out_element_num_ = output_element_num;
	layer.class_num_ = class_num;
	layer.layer_type_ = 'H';
	layer.map_size_ = GPUCNNDataStruct_Lib::RectSizePKG(1, 1);
	layer.out_map_num_ = output_element_num;
	return layer;

}
GPUCNNLayer GPUCNNLayer::CreateOutputLayer(size_t class_num)
{
	GPUCNNLayer layer;
	layer.class_num_ = class_num;
	layer.layer_type_ = 'O';
	layer.map_size_ = GPUCNNDataStruct_Lib::RectSizePKG(1, 1);
	layer.out_map_num_ = class_num;
	return layer;

}

void GPUCNNLayer::InitKernel(size_t front_map_num) {
	vec_kernel_.reserve(front_map_num * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_);
	vec_kernel_.resize(front_map_num * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_);
	size_t shift_idx_front_map = 0;
	size_t shift_idx_out_map = 0;

	for (size_t i = 0; i < front_map_num; i++)
	{
		shift_idx_front_map = i * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_;
		for (size_t j = 0; j < out_map_num_; j++)
		{
			shift_idx_out_map = j * kernel_size_.rows_ * kernel_size_.cols_;
			GPUAlgoOp_Lib::RandomMatrix(kernel_size_.rows_, kernel_size_.cols_, (vec_kernel_.data() + shift_idx_front_map + shift_idx_out_map));
		}
	}
}
//for adding momentum
void GPUCNNLayer::InitLastStepDeltaKernel(size_t front_map_num)
{
	vec_laststep_delta_kernel_.reserve(front_map_num * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_);
	vec_laststep_delta_kernel_.resize(front_map_num * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_);
	vec_laststep_delta_kernel_.assign(vec_laststep_delta_kernel_.size(), 0.0);

	size_t shift_idx_front_map = 0;
	size_t shift_idx_out_map = 0;

}
void GPUCNNLayer::InitOutputKernel(size_t front_map_num, GPUCNNDataStruct_Lib::RectSizePKG Kernel_size)
{
	kernel_size_ = Kernel_size;
	vec_kernel_.reserve(front_map_num * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_);
	vec_kernel_.resize(front_map_num * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_);
	size_t shift_idx_front_map = 0;
	size_t shift_idx_out_map = 0;

	for (size_t i = 0; i < front_map_num; i++)
	{
		shift_idx_front_map = i * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_;
		for (size_t j = 0; j < out_map_num_; j++)
		{
			shift_idx_out_map = j * kernel_size_.rows_ * kernel_size_.cols_;
			GPUAlgoOp_Lib::RandomMatrix(kernel_size_.rows_, kernel_size_.cols_, (vec_kernel_.data() + shift_idx_front_map + shift_idx_out_map));
		}
	}

}
//for adding momentum
void GPUCNNLayer::InitOutputLastStepDeltaKernel(size_t front_map_num, GPUCNNDataStruct_Lib::RectSizePKG Kernel_size)
{
	kernel_size_ = Kernel_size;
	vec_laststep_delta_kernel_.reserve(front_map_num * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_);
	vec_laststep_delta_kernel_.resize(front_map_num * out_map_num_ * kernel_size_.rows_ * kernel_size_.cols_);
	vec_laststep_delta_kernel_.assign(vec_laststep_delta_kernel_.size(), 0.0);
}
void GPUCNNLayer::InitErros(size_t batch_size)
{
	vec_errors_.reserve(batch_size * out_map_num_ * map_size_.rows_ * map_size_.cols_);
	vec_errors_.resize(batch_size * out_map_num_ * map_size_.rows_ * map_size_.cols_);
	vec_errors_.assign(vec_errors_.size(), 0.0);
}
void GPUCNNLayer::InitOutputMaps(size_t batch_size)
{
	vec_output_maps_.reserve(batch_size * out_map_num_ * map_size_.rows_ * map_size_.cols_);
	vec_output_maps_.resize(batch_size * out_map_num_ * map_size_.rows_ * map_size_.cols_);
	vec_output_maps_.assign(vec_output_maps_.size(), 0.0);
}
void GPUCNNLayer::InitBias(size_t front_map_num, size_t idx_iter)
{
	vec_bias_.reserve(out_map_num_);
	vec_bias_.resize(out_map_num_);
	vec_bias_.assign(vec_bias_.size(), 0.1);
}
void GPUCNNLayer::SetError(size_t num_batch, size_t map_no, size_t map_x, size_t map_y, float error_val)
{
	size_t shift_idx_error_batch_map = num_batch * out_map_num_ * map_size_.rows_ * map_size_.cols_;
	size_t shift_idx_error_out_map = map_no * map_size_.rows_ * map_size_.cols_;
	vec_errors_[shift_idx_error_batch_map + shift_idx_error_out_map + (map_x * map_size_.cols_) + map_y] = error_val;
}
void GPUCNNLayer::SetFCHLayerError(size_t num_batch, size_t map_no, float* p_matrix, size_t m, size_t n)
{
	size_t shift_idx_error_batch_map = num_batch * out_map_num_ * map_size_.rows_ * map_size_.cols_;
	size_t shift_idx_error_out_map = map_no * map_size_.rows_ * map_size_.cols_;
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			vec_errors_[shift_idx_error_batch_map + shift_idx_error_out_map + (i * map_size_.cols_) + j] = p_matrix[(i * n) + j];

		}
	}
}
void GPUCNNLayer::SetSampLayerError(size_t num_batch, size_t map_no, float* p_matrix, size_t m, size_t n)
{
	size_t shift_idx_error_batch_map = num_batch * out_map_num_ * map_size_.rows_ * map_size_.cols_;
	size_t shift_idx_error_out_map = map_no * map_size_.rows_ * map_size_.cols_;
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			vec_errors_[shift_idx_error_batch_map + shift_idx_error_out_map + (i * map_size_.cols_) + j] = p_matrix[(i * n) + j];

		}
	}
}
void GPUCNNLayer::SetConvLayerError(size_t num_batch, size_t map_no, float* p_matrix, size_t m, size_t n)
{
	size_t shift_idx_error_batch_map = num_batch * out_map_num_ * map_size_.rows_ * map_size_.cols_;
	size_t shift_idx_error_out_map = map_no * map_size_.rows_ * map_size_.cols_;
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			vec_errors_[shift_idx_error_batch_map + shift_idx_error_out_map + (i * map_size_.cols_) + j] = p_matrix[(i * n) + j];

		}
	}
}