/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description: CUDA ver.
******************************************************************************/
#include <algorithm>

#include "CUDACNNCls.cuh"
#include "CUDACNNLayer.cuh"

// CUDA_Algo_Lib::CUDACNN

#define DERIV_ACTIVE_RELU(S) 1 // derivative of the relu as a function of the relu's output
namespace CUDA_Algo_Lib
{
	size_t g_idx_epoch = 0;//index of epoch
	size_t g_idx_itor = 0;//index of iterator
	size_t g_idx_iter_init_bias = 0;//index of iterator for initialize bias
	size_t g_idx_iteration_num = 0;//index of iteration
	size_t g_iteration_num = 0;//number of g_iteration_num
}

void CUDA_Algo_Lib::CUDACNN::Train(CUDA_Algo_Lib::DatasetLoadingParamPKG& r_dataset_param)
{
	std::cout << "Start train" << std::endl;

	CUDA_Algo_Lib::g_iteration_num = r_dataset_param.total_num_images_ / batch_size_;
	if ((r_dataset_param.total_num_images_ % batch_size_) != 0)
	{
		std::cout << "Please reset CUDA_Algo_Lib::CUDACNN::batch_size_!" << std::endl;
	}

	float* p_train_batch_data = nullptr;
	float* p_train_batch_label = nullptr;
	std::vector<float> vec_train_batch_data;
	std::vector<float> vec_train_batch_label;
	vec_train_batch_data.reserve(batch_size_ * r_dataset_param.channels_image_ * r_dataset_param.rows_image_ * r_dataset_param.cols_image_);
	vec_train_batch_data.resize(batch_size_ * r_dataset_param.channels_image_ * r_dataset_param.rows_image_ * r_dataset_param.cols_image_);
	vec_train_batch_label.reserve(batch_size_ * r_dataset_param.num_output_cls_);
	vec_train_batch_label.resize(batch_size_ * r_dataset_param.num_output_cls_);

	for (CUDA_Algo_Lib::g_idx_iteration_num = 0; CUDA_Algo_Lib::g_idx_iteration_num < CUDA_Algo_Lib::g_iteration_num; CUDA_Algo_Lib::g_idx_iteration_num++)
	{
		std::cout << "NO.of iteration(training): " << CUDA_Algo_Lib::g_idx_iteration_num << std::endl;
		size_t idx_loaded_dataset_batch = CUDA_Algo_Lib::g_idx_iteration_num % (r_dataset_param.total_num_images_ / batch_size_);
		for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
		{
			std::cout << "NO.of batch(training): " << idx_batch << std::endl;

			std::vector<float>::iterator shift_begin_iter_loaded_dataset_batch_data;
			std::vector<float>::iterator shift_begin_iter_loaded_dataset_batch_label;
			shift_begin_iter_loaded_dataset_batch_data = r_dataset_param.vec_images_.begin() + (idx_loaded_dataset_batch * batch_size_ * r_dataset_param.channels_image_ * r_dataset_param.rows_image_ * r_dataset_param.cols_image_);
			shift_begin_iter_loaded_dataset_batch_label = r_dataset_param.vec_labels_.begin() + (idx_loaded_dataset_batch * batch_size_ * r_dataset_param.num_output_cls_);
			vec_train_batch_data.assign(shift_begin_iter_loaded_dataset_batch_data, (shift_begin_iter_loaded_dataset_batch_data + (batch_size_ * r_dataset_param.channels_image_ * r_dataset_param.rows_image_ * r_dataset_param.cols_image_)));
			vec_train_batch_label.assign(shift_begin_iter_loaded_dataset_batch_label, (shift_begin_iter_loaded_dataset_batch_label + (batch_size_ * r_dataset_param.num_output_cls_)));

		}


		Forward(vec_train_batch_data.data());
		BackPropagation(vec_train_batch_data.data(), vec_train_batch_label.data());
		UpdateParas();


	}
	std::cout << "Finish train" << std::endl;

}

void CUDA_Algo_Lib::CUDACNN::Setup(size_t batch_size)
{
	CUDA_Algo_Lib::VECCUDACNNLayers::iterator iter = vec_layers_.begin();

	(*iter).InitOutputMaps(batch_size);
	iter++;
	for (iter; iter < vec_layers_.end(); iter++)
	{
		CUDA_Algo_Lib::g_idx_iter_init_bias = CUDA_Algo_Lib::g_idx_iter_init_bias + 1;

		size_t frontMapNum = (*(iter - 1)).GetOutMapNum();

		switch ((*iter).GetType())
		{
		case 'I':
			break;
		case 'C':
			// set map RectSize
			(*iter).SetMapSize((*(iter - 1)).GetMapSize().Substract((*iter).GetKernelSize(), 1));
			// initial convolution kernel_, quantities: frontMapNum*outMapNum_
			(*iter).InitKernel(frontMapNum);
			(*iter).InitLastStepDeltaKernel(frontMapNum);//for adding momentum
			//each map has one bias_, so frontMapNum is not necessary
			(*iter).InitBias(frontMapNum, CUDA_Algo_Lib::g_idx_iter_init_bias);
			(*iter).InitErros(batch_size);
			// each r_layer should initialize output map
			(*iter).InitOutputMaps(batch_size);
			break;
		case 'S':
			(*iter).SetOutMapNum((frontMapNum));
			(*iter).SetMapSize((*(iter - 1)).GetMapSize().Divide((*iter).GetScaleSize()));
			(*iter).InitErros(batch_size);
			(*iter).InitOutputMaps(batch_size);
			break;
		case 'H':
			(*iter).InitOutputKernel(frontMapNum, (*(iter - 1)).GetMapSize());
			(*iter).InitOutputLastStepDeltaKernel(frontMapNum, (*(iter - 1)).GetMapSize());//for adding momentum			
			(*iter).InitBias(frontMapNum, CUDA_Algo_Lib::g_idx_iter_init_bias);
			(*iter).InitErros(batch_size);
			(*iter).InitOutputMaps(batch_size);
			break;
		case 'O':
			(*iter).InitOutputKernel(frontMapNum, (*(iter - 1)).GetMapSize());
			(*iter).InitOutputLastStepDeltaKernel(frontMapNum, (*(iter - 1)).GetMapSize());//for adding momentum
			(*iter).InitBias(frontMapNum, CUDA_Algo_Lib::g_idx_iter_init_bias);
			(*iter).InitErros(batch_size);
			(*iter).InitOutputMaps(batch_size);
			break;
		default:
			break;
		}
	}
}

void CUDA_Algo_Lib::CUDACNN::SetupTest(size_t batch_size)
{
	CUDA_Algo_Lib::VECCUDACNNLayers::iterator iter = vec_layers_.begin();

	(*iter).InitOutputMaps(batch_size);
	iter++;
	for (iter; iter < vec_layers_.end(); iter++)
	{
		CUDA_Algo_Lib::g_idx_iter_init_bias = CUDA_Algo_Lib::g_idx_iter_init_bias + 1;

		size_t frontMapNum = (*(iter - 1)).GetOutMapNum();

		switch ((*iter).GetType())
		{
		case 'I':
			break;
		case 'C':
			// set map RectSize
			(*iter).SetMapSize((*(iter - 1)).GetMapSize().Substract((*iter).GetKernelSize(), 1));
			// initial convolution kernel_, quantities: frontMapNum*outMapNum_
			(*iter).InitKernel(frontMapNum);

			break;

		default:
			break;
		}
	}
}

void CUDA_Algo_Lib::CUDACNN::BackPropagation(float* p_batch_data, float* p_batch_label)
{
	SetOutLayerErrors(p_batch_data, p_batch_label);
	SetHiddenLayerErrors();
}

void CUDA_Algo_Lib::CUDACNN::Forward(float* p_batch_data)
{
	SetInLayerOutput(p_batch_data);
	CUDA_Algo_Lib::VECCUDACNNLayers::iterator iter = vec_layers_.begin()+1;
	//iter++;
	for (iter; iter < vec_layers_.end(); iter++)
	{
		switch ((*iter).GetType())
		{
		case 'C':
			SetConvOutput((*iter), (*(iter - 1)));
			break;
		case 'S':
			SetSampOutput((*iter), (*(iter - 1)));
			break;
		case 'H':
			SetFCHLayerOutput((*iter), (*(iter - 1)));
			break;
		case 'O':
			SetOutLayerOutput((*iter), (*(iter - 1)));
			break;
		default:
			break;
		}

	}
}

void CUDA_Algo_Lib::CUDACNN::SetInLayerOutput(float* p_batch_data)
{
	std::cout << "Execute CUDA_Algo_Lib::CUDACNN::SetInLayerOutput()" << std::endl;

	CUDA_Algo_Lib::VECCUDACNNLayers::iterator iter = vec_layers_.begin();

	RectSize map_size = (*iter).GetMapSize();
	size_t out_map_num = (*iter).GetOutMapNum();

	std::copy(p_batch_data, (p_batch_data + (batch_size_ * out_map_num * map_size.rows_ * map_size.cols_)), (*iter).vec_output_maps_.begin());
}
// for change the value in m_Layers
void CUDA_Algo_Lib::CUDACNN::SetConvOutput(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer)
{
	std::cout << "Execute CUDA_Algo_Lib::CUDACNN::SetConvOutput()" << std::endl;
	
	size_t layer_map_num = r_layer.GetOutMapNum();
	size_t lastlayer_map_num = r_lastlayer.GetOutMapNum();
	size_t lastlayer_map_x = r_lastlayer.GetMapSize().rows_;
	size_t lastlayer_map_y = r_lastlayer.GetMapSize().cols_;
	size_t layer_kernel_x = r_layer.GetKernelSize().rows_;
	size_t layer_kernel_y = r_layer.GetKernelSize().cols_;
	size_t layer_map_x = r_layer.GetMapSize().rows_;
	size_t layer_map_y = r_layer.GetMapSize().cols_;
	std::vector<float> vec_sum(layer_map_x * layer_map_y, 0.0);
	std::vector<float> vec_sum_now(layer_map_x * layer_map_y, 0.0);

	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (size_t i = 0; i < layer_map_num; i++)
		{
			for (size_t j = 0; j < lastlayer_map_num; j++)
			{
				size_t shift_idx_lastlayer_batch_map = idx_batch * lastlayer_map_num * lastlayer_map_x * lastlayer_map_y;
				size_t shift_idx_lastlayer_out_map = j * lastlayer_map_x * lastlayer_map_y;
				float* p_lastlayer_map = r_lastlayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;
				//float** lastMap;
				//lastMap = r_lastlayer.outputmaps_[idx_batch][j];				
				size_t shift_idx_layer_front_kernel = j * layer_map_num * layer_kernel_x * layer_kernel_y;
				size_t shift_idx_layer_out_kernel = i * layer_kernel_x * layer_kernel_y;
				float* p_layer_kernel = r_layer.vec_kernel_.data() + shift_idx_layer_front_kernel + shift_idx_layer_out_kernel;

				if (j == 0)
				{
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum.data());
					//each time we calculate one image of batch and also calculate relu 

				}
				else {
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum_now.data());
					CalConvArrayPlus(vec_sum_now.data(), vec_sum.data(), layer_map_x, layer_map_y);// sumNow 

				}
			}

			ActiveRelu(vec_sum.data(), r_layer.vec_bias_.at(i), layer_map_x, layer_map_y);//for relu active fun.

			size_t shift_idx_layer_batch_map = idx_batch * layer_map_num * layer_map_x * layer_map_y;
			size_t shift_idx_layer_out_map = i * layer_map_x * layer_map_y;
			float* p_layer_out_map = r_layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;
			memcpy(p_layer_out_map, vec_sum.data(), (layer_map_x * layer_map_y * sizeof(float)));

		}
	}

}

void CUDA_Algo_Lib::CUDACNN::SetSampOutput(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer)
{
	std::cout << "Execute CUDA_Algo_Lib::CUDACNN::SetSampOutput()" << std::endl;

	size_t lastlayer_map_num = r_lastlayer.GetOutMapNum();
	size_t lastlayer_map_x = r_lastlayer.GetMapSize().rows_;
	size_t lastlayer_map_y = r_lastlayer.GetMapSize().cols_;
	size_t layer_map_x = r_layer.GetMapSize().rows_;
	size_t layer_map_y = r_layer.GetMapSize().cols_;
	RectSize scale_size = r_layer.GetScaleSize();
	std::vector<float> vec_samp_matrix(layer_map_x*layer_map_y, 0.0);

	float* p_lastlayer_map = NULL;
	size_t shift_idx_lastlayer_batch_map = 0;
	size_t shift_idx_lastlayer_out_map = 0;

	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (size_t i = 0; i < lastlayer_map_num; i++)
		{
			shift_idx_lastlayer_batch_map = idx_batch * lastlayer_map_num * lastlayer_map_x * lastlayer_map_y;
			shift_idx_lastlayer_out_map = i * lastlayer_map_x * lastlayer_map_y;
			p_lastlayer_map = r_lastlayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;			
			ScaleMatrix(p_lastlayer_map, scale_size, lastlayer_map_x, lastlayer_map_y, vec_samp_matrix.data());
			
			size_t shift_idx_layer_batch_map = idx_batch * lastlayer_map_num * layer_map_x * layer_map_y;
			size_t shift_idx_layer_out_map = i * layer_map_x * layer_map_y;
			float* p_layer_out_map = r_layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;
			memcpy(p_layer_out_map, vec_samp_matrix.data(), (layer_map_x * layer_map_y * sizeof(float)));
		}
	}
}

void CUDA_Algo_Lib::CUDACNN::SetFCHLayerOutput(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer)
{
	std::cout << "Execute CUDA_Algo_Lib::CUDACNN::SetFCHLayerOutput()" << std::endl;

	size_t layer_map_num = r_layer.GetOutMapNum();
	size_t lastlayer_map_num = r_lastlayer.GetOutMapNum();
	size_t lastlayer_map_x = r_lastlayer.GetMapSize().rows_;
	size_t lastlayer_map_y = r_lastlayer.GetMapSize().cols_;
	size_t layer_kernel_x = r_layer.GetKernelSize().rows_;
	size_t layer_kernel_y = r_layer.GetKernelSize().cols_;
	size_t layer_map_x = r_layer.GetMapSize().rows_;
	size_t layer_map_y = r_layer.GetMapSize().cols_;
	std::vector<float> vec_sum(layer_map_x * layer_map_y, 0.0);
	std::vector<float> vec_sum_now(layer_map_x * layer_map_y, 0.0);

	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (size_t i = 0; i < layer_map_num; i++)
		{
			for (size_t j = 0; j < lastlayer_map_num; j++)
			{
				size_t shift_idx_lastlayer_batch_map = idx_batch * lastlayer_map_num * lastlayer_map_x * lastlayer_map_y;
				size_t shift_idx_lastlayer_out_map = j * lastlayer_map_x * lastlayer_map_y;
				float* p_lastlayer_map = r_lastlayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;
			
				size_t shift_idx_layer_front_kernel = j * layer_map_num * layer_kernel_x * layer_kernel_y;
				size_t shift_idx_layer_out_kernel = i * layer_kernel_x * layer_kernel_y;
				float* p_layer_kernel = r_layer.vec_kernel_.data() + shift_idx_layer_front_kernel + shift_idx_layer_out_kernel;

				if (j == 0)
				{
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum.data());
					//each time we calculate one image of batch and also calculate relu 

				}
				else {
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum_now.data());
					CalFCHArrayPlus(vec_sum_now.data(), vec_sum.data(), layer_map_x, layer_map_y);// sumNow 

				}
			}

			//printf("ActiveRelu");
			ActiveRelu(vec_sum.data(), r_layer.vec_bias_.at(i), layer_map_x, layer_map_y);//for relu active fun.

			//SetValue(r_layer.outputmaps_[idx_batch][i], sum, layer_map_x, layer_map_y);
			size_t shift_idx_layer_batch_map = idx_batch * layer_map_num * layer_map_x * layer_map_y;
			size_t shift_idx_layer_out_map = i * layer_map_x * layer_map_y;
			float* p_layer_out_map = r_layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;
			memcpy(p_layer_out_map, vec_sum.data(), (layer_map_x * layer_map_y * sizeof(float)));

		}

	}
}

// ReLU+Softmax function
void CUDA_Algo_Lib::CUDACNN::SetOutLayerOutput(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer)
{
	std::cout << "Execute CUDA_Algo_Lib::CUDACNN::SetOutLayerOutput()" << std::endl;

	size_t layer_map_num = r_layer.GetOutMapNum();
	size_t lastlayer_map_num = r_lastlayer.GetOutMapNum();
	size_t lastlayer_map_x = r_lastlayer.GetMapSize().rows_;
	size_t lastlayer_map_y = r_lastlayer.GetMapSize().cols_;
	size_t layer_kernel_x = r_layer.GetKernelSize().rows_;
	size_t layer_kernel_y = r_layer.GetKernelSize().cols_;
	size_t layer_map_x = r_layer.GetMapSize().rows_;
	size_t layer_map_y = r_layer.GetMapSize().cols_;
	std::vector<float> vec_sum(layer_map_x * layer_map_y, 0.0);
	std::vector<float> vec_sum_now(layer_map_x * layer_map_y, 0.0);
	std::vector<float> vec_sum_expone(batch_size_, 0.0);

	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		//printf("ActiveRelu+softmax");
		//std::cout << "NO.of Batch: " << idx_batch << std::endl;
		for (size_t i = 0; i < layer_map_num; i++)
		{
			for (size_t j = 0; j < lastlayer_map_num; j++)
			{
				size_t shift_idx_lastlayer_batch_map = idx_batch * lastlayer_map_num * lastlayer_map_x * lastlayer_map_y;
				size_t shift_idx_lastlayer_out_map = j * lastlayer_map_x * lastlayer_map_y;
				float* p_lastlayer_map = r_lastlayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;
			
				size_t shift_idx_layer_front_kernel = j * layer_map_num * layer_kernel_x * layer_kernel_y;
				size_t shift_idx_layer_out_kernel = i * layer_kernel_x * layer_kernel_y;
				float* p_layer_kernel = r_layer.vec_kernel_.data() + shift_idx_layer_front_kernel + shift_idx_layer_out_kernel;

				if (j == 0)
				{
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum.data());
					//each time we calculate one image of batch and also calculate relu 

				}
				else {
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum_now.data());
					CalFCHArrayPlus(vec_sum_now.data(), vec_sum.data(), layer_map_x, layer_map_y);// sumNow 

				}
			}

			CalExpone(vec_sum.data(), r_layer.vec_bias_.at(i), layer_map_x, layer_map_y);

			size_t shift_idx_layer_batch_map = idx_batch * layer_map_num * layer_map_x * layer_map_y;
			size_t shift_idx_layer_out_map = i * layer_map_x * layer_map_y;
			float* p_layer_out_map = r_layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;
			memcpy(p_layer_out_map, vec_sum.data(), (layer_map_x * layer_map_y * sizeof(float)));

		}

		for (size_t i = 0; i < layer_map_num; i++)
		{
			for (size_t ii = 0; ii < layer_map_x; ii++)
			{
				for (size_t jj = 0; jj < layer_map_y; jj++)
				{
					size_t shift_idx_layer_batch_map = idx_batch * layer_map_num * layer_map_x * layer_map_y;
					size_t shift_idx_layer_out_map = i * layer_map_x * layer_map_y;
					size_t shift_idx_layer_out_map_row = ii * layer_map_y;
					size_t idx_layer_out_map = shift_idx_layer_batch_map + shift_idx_layer_out_map + shift_idx_layer_out_map_row + jj;
					vec_sum_expone[idx_batch] += r_layer.vec_output_maps_.at(idx_layer_out_map);
				}
			}
		}

		for (size_t i = 0; i < layer_map_num; i++)
		{
			for (size_t ii = 0; ii < layer_map_x; ii++)
			{
				for (size_t jj = 0; jj < layer_map_y; jj++)
				{

					size_t shift_idx_layer_batch_map = idx_batch * layer_map_num * layer_map_x * layer_map_y;
					size_t shift_idx_layer_out_map = i * layer_map_x * layer_map_y;
					size_t shift_idx_layer_out_map_row = ii * layer_map_y;
					size_t idx_layer_out_map = shift_idx_layer_batch_map + shift_idx_layer_out_map + shift_idx_layer_out_map_row + jj;
					r_layer.vec_output_maps_[idx_layer_out_map] = r_layer.vec_output_maps_[idx_layer_out_map] / vec_sum_expone[idx_batch];
					
					std::cout << "Outputlayer's Softmax actual output(r_layer.outputmaps_[" << idx_batch << "][" << i << "][" << ii << "][" << jj << "]): " << r_layer.vec_output_maps_[idx_layer_out_map] << std::endl;
				}
			}
		}
	}

}

void CUDA_Algo_Lib::CUDACNN::SetOutLayerErrors(float* p_input_maps, float* p_target_labels)
{
	CUDA_Algo_Lib::VECCUDACNNLayers::iterator iter = vec_layers_.end();
	iter--;
	size_t layer_outmap_num = (*iter).GetOutMapNum();
	float mean_error = 0.0, max_error = 0.0;

	//FILE* fy;
	//fy = fopen("./outputdata/error.txt", "a");

	////if( (err=fopen_s(&fy, "error.txt", "a")) != 0 )
	////	exit(1) ;

	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (size_t idx_map = 0; idx_map < layer_outmap_num; idx_map++)
		{
			//float val_out_map = (*iter).outputmaps_[idx_batch][idx_map][0][0];
			float val_target_label = p_target_labels[idx_batch * layer_outmap_num + idx_map];
			size_t shift_idx_layer_batch_map = idx_batch * layer_outmap_num * ((*iter).GetMapSize().rows_) * ((*iter).GetMapSize().cols_);
			size_t shift_idx_layer_out_map = idx_map * ((*iter).GetMapSize().rows_) * ((*iter).GetMapSize().cols_);
			size_t shift_idx_layer_out_map_row = 0 * ((*iter).GetMapSize().cols_);
			size_t idx_layer_out_map = shift_idx_layer_batch_map + shift_idx_layer_out_map + shift_idx_layer_out_map_row + 0;
			float val_out_map = (*iter).vec_output_maps_.at(idx_layer_out_map);

			//printf("Cross-entropy cost function for ReLU+Softmax");
			//Cross entropy for softmax form
			(*iter).SetError(idx_batch, idx_map, 0, 0, (val_target_label - val_out_map));
			mean_error = abs(val_target_label - val_out_map);

			//fprintf(fy, "%f ", mean_error);
			//// 			mean_error += abs(val_target_label-val_out_map);
			//// 			if (abs(val_target_label-val_out_map)>max_error)
			//// 			{
			//// 				max_error = abs(val_target_label-val_out_map);
			//// 			}
		}
		//fprintf(fy, "\n");
	}
	//fprintf(fy, "\n");
	//fclose(fy);
	//// 	std::cout<<"Mean error of each mini batch: "<<mean_error<<std::endl;
	//// 	std::cout<<"The max error of one output in mini batch: "<<max_error<<std::endl;
}

void CUDA_Algo_Lib::CUDACNN::SetFCHiddenLayerErrors(CUDA_Algo_Lib::CUDACNNLayer& Lastlayer, CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_nextlayer)//for add FC hiddenlayer
{
	size_t lastlayer_outmap_num = Lastlayer.GetOutMapNum();
	size_t layer_outmap_num = r_layer.GetOutMapNum();
	size_t layer_outmap_rows = r_layer.GetMapSize().rows_;
	size_t layer_outmap_cols = r_layer.GetMapSize().cols_;
	size_t nextlayer_outmap_num = r_nextlayer.GetOutMapNum();
	size_t nextlayer_outmap_rows = r_nextlayer.GetMapSize().rows_;
	size_t nextlayer_outmap_cols = r_nextlayer.GetMapSize().cols_;

	float* p_layer_outmap = nullptr;
	float* p_layer_error = nullptr;
	float* p_nextlayer_error = nullptr;
	std::vector<float> vec_layer_outmatrix;
	std::vector<float> vec_layer_outkroneckermatrix;
	vec_layer_outmatrix.reserve(layer_outmap_rows * layer_outmap_cols);
	vec_layer_outmatrix.resize(layer_outmap_rows * layer_outmap_cols);
	vec_layer_outkroneckermatrix.reserve(layer_outmap_rows * layer_outmap_cols);
	vec_layer_outkroneckermatrix.resize(layer_outmap_rows * layer_outmap_cols);
	RectSize layer_scale_size = r_layer.GetScaleSize();

	size_t nextlayer_kernel_rows = r_nextlayer.GetKernelSize().rows_;
	size_t nextlayer_kernel_cols = r_nextlayer.GetKernelSize().cols_;

	float* p_nextlayer_kernel = nullptr;

	std::vector<float> vec_derivative_active_fun;
	vec_derivative_active_fun.reserve(batch_size_ * layer_outmap_num);
	vec_derivative_active_fun.resize(batch_size_ * layer_outmap_num);

	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (size_t idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			size_t shift_idx_layer_batch_map = idx_batch * layer_outmap_num * layer_outmap_rows * layer_outmap_cols;
			size_t shift_idx_layer_out_map = idx_layer_outmap * layer_outmap_rows * layer_outmap_cols;
			p_layer_outmap = r_layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;
			
			r_layer.SetFCHLayerError(idx_batch, idx_layer_outmap, p_layer_outmap, layer_outmap_rows, layer_outmap_cols);

			//printf("derivative of ReLu");
			float* p_derivative_active_fun = vec_derivative_active_fun.data() + (idx_batch * layer_outmap_num + idx_layer_outmap);
			MatrixDreluFChidden(p_layer_outmap, layer_outmap_rows, layer_outmap_cols, p_derivative_active_fun);//for relu active fun.

		}
	}
	//printf("================================================================================\n");

	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (size_t idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			p_layer_error = r_layer.GetError(idx_batch, idx_layer_outmap);
			p_layer_error[0 * layer_outmap_num + 0] = vec_derivative_active_fun.at(idx_batch * layer_outmap_num + idx_layer_outmap);

		}
	}
	//printf("================================================================================\n");

	std::vector<float> vec_sum_local_gradient((batch_size_ * layer_outmap_num), 0.0);

	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (size_t idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			for (size_t idx_nextlayer_outmap = 0; idx_nextlayer_outmap < nextlayer_outmap_num; idx_nextlayer_outmap++)
			{
				p_layer_error = r_nextlayer.GetError(idx_batch, idx_nextlayer_outmap);
				p_nextlayer_kernel = r_nextlayer.GetKernel(idx_layer_outmap, idx_nextlayer_outmap);

				vec_sum_local_gradient[idx_batch * layer_outmap_num + idx_layer_outmap] += p_layer_error[0 * nextlayer_outmap_num + 0] * p_nextlayer_kernel[0 * nextlayer_kernel_cols + 0];

			}

		}
	}

	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (size_t idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			p_layer_error = r_layer.GetError(idx_batch, idx_layer_outmap);
			if (0.0 == p_layer_error[0 * nextlayer_outmap_num + 0])
			{
				p_layer_error[0 * nextlayer_outmap_num + 0] = p_layer_error[0 * nextlayer_outmap_num + 0] * vec_sum_local_gradient[idx_batch * layer_outmap_num + idx_layer_outmap];

				p_layer_error[0 * nextlayer_outmap_num + 0] = abs(p_layer_error[0 * nextlayer_outmap_num + 0]);
			}
			else {
				p_layer_error[0 * nextlayer_outmap_num + 0] = p_layer_error[0 * nextlayer_outmap_num + 0] * vec_sum_local_gradient[idx_batch * layer_outmap_num + idx_layer_outmap];
			}

			r_layer.SetFCHLayerError(idx_batch, idx_layer_outmap, p_layer_error, 0, 0);

		}
	}

}

void CUDA_Algo_Lib::CUDACNN::SetHiddenLayerErrors()
{
	CUDA_Algo_Lib::VECCUDACNNLayers::iterator iter = vec_layers_.end();
	iter = iter - 2;
	for (iter; iter > vec_layers_.begin(); iter--)
	{
		switch ((*(iter)).GetType())
		{
		case 'C':
			SetConvErrors((*iter), (*(iter + 1)));
			break;
		case 'S':
			SetSampErrors((*iter), (*(iter + 1)));
			break;
		case 'H':
			SetFCHiddenLayerErrors((*(iter - 1)), (*iter), (*(iter + 1)));
			break;
		default:
			break;
		}
	}
}

void CUDA_Algo_Lib::CUDACNN::SetSampErrors(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_nextlayer)
{
	size_t layer_outmap_num = r_layer.GetOutMapNum();
	size_t layer_outmap_rows = r_layer.GetMapSize().rows_;
	size_t layer_outmap_cols = r_layer.GetMapSize().cols_;
	size_t nextlayer_outmap_num = r_nextlayer.GetOutMapNum();
	size_t nextlayer_outmap_rows = r_nextlayer.GetMapSize().rows_;
	size_t nextlayer_outmap_cols = r_nextlayer.GetMapSize().cols_;
	size_t nextlayer_kernel_rows = r_nextlayer.GetKernelSize().rows_;
	size_t nextlayer_kernel_cols = r_nextlayer.GetKernelSize().cols_;

	float* p_nextlayer_error = nullptr;
	float* p_nextlayer_kernel = nullptr;
	std::vector<float> vec_sum(layer_outmap_rows * layer_outmap_cols, 0.0);
	std::vector<float> vec_sum_now(layer_outmap_rows * layer_outmap_cols, 0.0);
	std::vector<float> vec_rot_matrix(nextlayer_kernel_rows * nextlayer_kernel_cols, 0.0);
	std::vector<float> vec_nextlayer_extend_matrix((nextlayer_outmap_rows+2*(nextlayer_kernel_rows-1)) * (nextlayer_outmap_cols+2*(nextlayer_kernel_cols-1)), 0.0);

	//calculate
	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (size_t idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			for (size_t idx_nextlayer_outmap = 0; idx_nextlayer_outmap < nextlayer_outmap_num; idx_nextlayer_outmap++)
			{

				p_nextlayer_error = r_nextlayer.GetError(idx_batch, idx_nextlayer_outmap);
				p_nextlayer_kernel = r_nextlayer.GetKernel(idx_layer_outmap, idx_nextlayer_outmap);
				if (idx_nextlayer_outmap == 0)
				{
					Rot180(p_nextlayer_kernel, nextlayer_kernel_rows, nextlayer_kernel_cols, vec_rot_matrix.data());
					ConvNSampFull(p_nextlayer_error, vec_rot_matrix.data(), nextlayer_outmap_rows, nextlayer_outmap_cols, nextlayer_kernel_rows, nextlayer_kernel_cols, vec_sum.data(), vec_nextlayer_extend_matrix.data());

				}
				else
				{
					Rot180(p_nextlayer_kernel, nextlayer_kernel_rows, nextlayer_kernel_cols, vec_rot_matrix.data());
					ConvNSampFull(p_nextlayer_error, vec_rot_matrix.data(), nextlayer_outmap_rows, nextlayer_outmap_cols, nextlayer_kernel_rows, nextlayer_kernel_cols, vec_sum_now.data(), vec_nextlayer_extend_matrix.data());
					CalSampArrayPlus(vec_sum_now.data(), vec_sum.data(), layer_outmap_rows, layer_outmap_cols);

				}

			}
			r_layer.SetSampLayerError(idx_batch, idx_layer_outmap, vec_sum.data(), layer_outmap_rows, layer_outmap_cols);
		}
	}

}

void CUDA_Algo_Lib::CUDACNN::SetConvErrors(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_nextlayer)
{
	size_t layer_outmap_num = r_layer.GetOutMapNum();
	size_t layer_outmap_rows = r_layer.GetMapSize().rows_;
	size_t layer_outmap_cols = r_layer.GetMapSize().cols_;
	size_t nextlayer_outmap_rows = r_nextlayer.GetMapSize().rows_;
	size_t nextlayer_outmap_cols = r_nextlayer.GetMapSize().cols_;

	float* p_nextlayer_error = nullptr;
	float* p_layer_outmap = nullptr;
	std::vector<float> vec_layer_outmatrix;
	std::vector<float> vec_layer_outkroneckermatrix;
	vec_layer_outmatrix.reserve(layer_outmap_rows * layer_outmap_cols);
	vec_layer_outkroneckermatrix.reserve(layer_outmap_rows * layer_outmap_cols);
	vec_layer_outmatrix.resize(layer_outmap_rows * layer_outmap_cols);
	vec_layer_outkroneckermatrix.resize(layer_outmap_rows * layer_outmap_cols);
	RectSize layer_scale_size = r_layer.GetScaleSize();

	for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (size_t idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			layer_scale_size = r_nextlayer.GetScaleSize();
			p_nextlayer_error = r_nextlayer.GetError(idx_batch, idx_layer_outmap);
			size_t shift_idx_layer_batch_map = idx_batch * layer_outmap_num * layer_outmap_rows * layer_outmap_cols;
			size_t shift_idx_layer_out_map = idx_layer_outmap * layer_outmap_rows * layer_outmap_cols;
			p_layer_outmap = r_layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;

			//printf("derivative of ReLu");
			//derivative of ReLu
			MatrixDreluConv(p_layer_outmap, layer_outmap_rows, layer_outmap_cols, vec_layer_outmatrix.data());//for relu active fun.

			CalKronecker(p_nextlayer_error, layer_scale_size, nextlayer_outmap_rows, nextlayer_outmap_cols, vec_layer_outkroneckermatrix.data(), layer_outmap_rows, layer_outmap_cols);
			CalMatrixMultiply(vec_layer_outmatrix.data(), vec_layer_outkroneckermatrix.data(), layer_outmap_rows, layer_outmap_cols);

			r_layer.SetConvLayerError(idx_batch, idx_layer_outmap, vec_layer_outmatrix.data(), layer_outmap_rows, layer_outmap_cols);

		}
	}
}

void CUDA_Algo_Lib::CUDACNN::UpdateParas()
{
	CUDA_Algo_Lib::VECCUDACNNLayers::iterator iter = vec_layers_.begin();
	iter++;

	CUDA_Algo_Lib::g_idx_itor = 0;//begining at index 0 r_layer
	char str_file_kernel[1000];// initialized properly
	char str_file_bias[1000];// initialized properly

	for (iter; iter < vec_layers_.end(); iter++)
	{
		CUDA_Algo_Lib::g_idx_itor = CUDA_Algo_Lib::g_idx_itor + 1;
		sprintf(str_file_kernel, "./data/kernel_weight/kernel_weight_%d_%d", CUDA_Algo_Lib::g_idx_itor, (*iter).GetType());
		sprintf(str_file_bias, "./data/bias/bias_%d_%d", CUDA_Algo_Lib::g_idx_itor, (*iter).GetType());
		//printf("%s", str_file_kernel);

		switch ((*iter).GetType())
		{
		case 'C':
			UpdateKernels(*iter, *(iter - 1), str_file_kernel, eta_conv_, alpha_conv_);
			UpdateBias(*iter, str_file_bias, eta_conv_);
			break;
		case 'H':
			UpdateKernels(*iter, *(iter - 1), str_file_kernel, eta_fc_, alpha_fc_);
			UpdateBias(*iter, str_file_bias, eta_fc_);
			break;
		case 'O':
			UpdateKernels(*iter, *(iter - 1), str_file_kernel, eta_fc_, alpha_fc_);
			UpdateBias(*iter, str_file_bias, eta_fc_);
			break;
		default:
			break;
		}
	}
}

void CUDA_Algo_Lib::CUDACNN::UpdateBias(CUDA_Algo_Lib::CUDACNNLayer& r_layer, char* str_File_Bias, float eta)
{
	size_t layer_outmap_num = r_layer.GetOutMapNum();
	size_t layer_outmap_rows = r_layer.GetMapSize().rows_;
	size_t layer_outmap_cols = r_layer.GetMapSize().cols_;
	float* p_layer_error = r_layer.vec_errors_.data();
	std::vector<float> vec_error((layer_outmap_rows* layer_outmap_cols), 0.0);
	float deltaBias = 0.0;

	for (size_t idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
	{

		CalErrorsSum(p_layer_error, idx_layer_outmap, layer_outmap_num, layer_outmap_rows, layer_outmap_cols, batch_size_, vec_error.data());
		deltaBias = (CalErrorSum(vec_error.data(), layer_outmap_rows, layer_outmap_cols) / ((float)batch_size_));
		r_layer.vec_bias_.at(idx_layer_outmap) += (eta * deltaBias);

		/***save bias_***/
		if ((CUDA_Algo_Lib::g_iteration_num - 1) == CUDA_Algo_Lib::g_idx_iteration_num) {
			char str_file_bias_1[1000];
			sprintf(str_file_bias_1, "%s_%d.txt", str_File_Bias, idx_layer_outmap);
			FILE* fp_bias = fopen(str_file_bias_1, "w");

			fprintf(fp_bias, "%f ", r_layer.vec_bias_.at(idx_layer_outmap));
			fprintf(fp_bias, "\n");

			fclose(fp_bias);
		}
	}

}

void CUDA_Algo_Lib::CUDACNN::UpdateKernels(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer, char* str_File_Kernel, float eta, float alpha)
{
	size_t lastlayer_outmap_num = r_lastlayer.GetOutMapNum();
	size_t lastlayer_outmap_rows = r_lastlayer.GetMapSize().rows_;
	size_t lastlayer_outmap_cols = r_lastlayer.GetMapSize().cols_;
	size_t layer_outmap_num = r_layer.GetOutMapNum();
	size_t layer_outmap_rows = r_layer.GetMapSize().rows_;
	size_t layer_outmap_cols = r_layer.GetMapSize().cols_;
	size_t layer_kernel_rows = r_layer.GetKernelSize().rows_;
	size_t layer_kernel_cols = r_layer.GetKernelSize().cols_;


	std::vector<float> vec_delta_kernel_1((layer_kernel_rows * layer_kernel_cols), 0.0);
	std::vector<float> vec_delta_kernel_2((layer_kernel_rows * layer_kernel_cols), 0.0);
	std::vector<float> vec_delta_now((layer_kernel_rows * layer_kernel_cols), 0.0);
	float* p_layer_error = nullptr;
	float* p_lastlayer_outmap = nullptr;
	float* p_layer_laststep_delta_kernel = nullptr;
	float* p_layer_kernel = nullptr;

	for (size_t idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
	{
		for (size_t idx_lastlayer_outmap = 0; idx_lastlayer_outmap < lastlayer_outmap_num; idx_lastlayer_outmap++)
		{
			for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
			{
				p_layer_error = r_layer.GetError(idx_batch, idx_layer_outmap);
				if (idx_batch == 0) {
					size_t shift_idx_lastlayer_batch_map = idx_batch * lastlayer_outmap_num * lastlayer_outmap_rows * lastlayer_outmap_cols;
					size_t shift_idx_lastlayer_out_map = idx_lastlayer_outmap * lastlayer_outmap_rows * lastlayer_outmap_cols;
					p_lastlayer_outmap = r_lastlayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;
					ConvNValid(p_lastlayer_outmap, p_layer_error, lastlayer_outmap_rows, lastlayer_outmap_cols, layer_outmap_rows, layer_outmap_cols, vec_delta_kernel_1.data());
				}
				else {
					size_t shift_idx_lastlayer_batch_map = idx_batch * lastlayer_outmap_num * lastlayer_outmap_rows * lastlayer_outmap_cols;
					size_t shift_idx_lastlayer_out_map = idx_lastlayer_outmap * lastlayer_outmap_rows * lastlayer_outmap_cols;
					p_lastlayer_outmap = r_lastlayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;
					ConvNValid(p_lastlayer_outmap, p_layer_error, lastlayer_outmap_rows, lastlayer_outmap_cols, layer_outmap_rows, layer_outmap_cols, vec_delta_now.data());
					CalConvArrayPlus(vec_delta_now.data(), vec_delta_kernel_1.data(), layer_kernel_rows, layer_kernel_cols);
				}
			}
			size_t shift_idx_layer_kernel_lastlayer = idx_lastlayer_outmap * layer_outmap_num * layer_kernel_rows * layer_kernel_cols;
			size_t shift_idx_layer_kernel_layer = idx_layer_outmap * layer_kernel_rows * layer_kernel_cols;
			p_layer_laststep_delta_kernel = r_layer.vec_laststep_delta_kernel_.data() + shift_idx_layer_kernel_lastlayer + shift_idx_layer_kernel_layer;
			p_layer_kernel = r_layer.vec_kernel_.data() + shift_idx_layer_kernel_lastlayer + shift_idx_layer_kernel_layer;
			SetKernelValue(vec_delta_kernel_2.data(), p_layer_laststep_delta_kernel, layer_kernel_rows, layer_kernel_cols);
			CalArrayMultiply(vec_delta_kernel_2.data(), alpha, layer_kernel_rows, layer_kernel_cols);//for adding momentum
			CalArrayPlus(vec_delta_kernel_2.data(), p_layer_kernel, layer_kernel_rows, layer_kernel_cols);//for adding momentum
			CalArrayDivide(vec_delta_kernel_1.data(), batch_size_, layer_kernel_rows, layer_kernel_cols);
			CalArrayMultiply(vec_delta_kernel_1.data(), eta, layer_kernel_rows, layer_kernel_cols);
			SetKernelValue(p_layer_laststep_delta_kernel, vec_delta_kernel_1.data(), layer_kernel_rows, layer_kernel_cols);//for adding momentum
			CalArrayPlus(vec_delta_kernel_1.data(), p_layer_kernel, layer_kernel_rows, layer_kernel_cols);

			/***save kernel_ weight***/
			if ((CUDA_Algo_Lib::g_iteration_num - 1) == CUDA_Algo_Lib::g_idx_iteration_num) {
				char str_file_kernel_1[1000];
				sprintf(str_file_kernel_1, "%s_%d_%d.txt", str_File_Kernel, idx_lastlayer_outmap, idx_layer_outmap);

				FILE* fp = fopen(str_file_kernel_1, "w");
				size_t shift_idx_layer_kernel_lastlayer = 0;
				size_t shift_idx_layer_kernel_layer = 0;
				size_t idx_layer_kernel = 0;

				for (size_t mm = 0; mm < layer_kernel_rows; mm++)
				{
					for (size_t nn = 0; nn < layer_kernel_cols; nn++)
					{
						shift_idx_layer_kernel_lastlayer = idx_lastlayer_outmap * layer_outmap_num * layer_kernel_rows * layer_kernel_cols;
						shift_idx_layer_kernel_layer = idx_layer_outmap * layer_kernel_rows * layer_kernel_cols;
						idx_layer_kernel = shift_idx_layer_kernel_lastlayer + shift_idx_layer_kernel_layer + (mm * layer_kernel_cols + nn);

						fprintf(fp, "%f ", r_layer.vec_kernel_.at(idx_layer_kernel));
					}

				}
				fprintf(fp, "\n");
				fclose(fp);
			}

		}
	}

}

void CUDA_Algo_Lib::CUDACNN::LoadParas()
{
	CUDA_Algo_Lib::VECCUDACNNLayers::iterator iter = vec_layers_.begin();
	iter++;

	CUDA_Algo_Lib::g_idx_itor = 0;//begining at index 0 r_layer
	char str_file_kernel[1000];// initialized properly
	char str_file_bias[1000];// initialized properly

	for (iter; iter < vec_layers_.end(); iter++)
	{
		CUDA_Algo_Lib::g_idx_itor = CUDA_Algo_Lib::g_idx_itor + 1;
		sprintf(str_file_kernel, "./data/kernel_weight/kernel_weight_%d_%d", CUDA_Algo_Lib::g_idx_itor, (*iter).GetType());
		sprintf(str_file_bias, "./data/bias/bias_%d_%d", CUDA_Algo_Lib::g_idx_itor, (*iter).GetType());
		//printf("%s", str_file_kernel);

		switch ((*iter).GetType())
		{
		case 'C':
			LoadKernels(*iter, *(iter - 1), str_file_kernel);
			LoadBias(*iter, str_file_bias);
			break;
		case 'H':
			LoadKernels(*iter, *(iter - 1), str_file_kernel);
			LoadBias(*iter, str_file_bias);
			break;
		case 'O':
			LoadKernels(*iter, *(iter - 1), str_file_kernel);
			LoadBias(*iter, str_file_bias);
			break;
		default:
			break;
		}
	}
}

void CUDA_Algo_Lib::CUDACNN::LoadBias(CUDA_Algo_Lib::CUDACNNLayer& r_layer, char* str_File_Bias)
{
	size_t layer_outmap_num = r_layer.GetOutMapNum();
	float bias = 0.0;

	for (size_t idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
	{
		bias = 0.0;
		/***load bias***/
		char str_file_bias_1[1000];
		sprintf(str_file_bias_1, "%s_%d.txt", str_File_Bias, idx_layer_outmap);
		printf("%s\n", str_file_bias_1);
		FILE* fp_bias = fopen(str_file_bias_1, "r");
		fscanf(fp_bias, "%f ", &bias);
		fclose(fp_bias);

		r_layer.vec_bias_.at(idx_layer_outmap) = bias;
		printf("bias: %f\n", r_layer.vec_bias_.at(idx_layer_outmap));
	}
	
}

void CUDA_Algo_Lib::CUDACNN::LoadKernels(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer, char* str_File_Kernel)
{
	
	const size_t lastlayer_outmap_num = r_lastlayer.GetOutMapNum();
	const size_t lastlayer_outmap_rows = r_lastlayer.GetMapSize().rows_;
	const size_t lastlayer_outmap_cols = r_lastlayer.GetMapSize().cols_;
	const size_t layer_outmap_num = r_layer.GetOutMapNum();
	const size_t layer_outmap_rows = r_layer.GetMapSize().rows_;
	const size_t layer_outmap_cols = r_layer.GetMapSize().cols_;
	const size_t layer_kernel_rows = r_layer.GetKernelSize().rows_;
	const size_t layer_kernel_cols = r_layer.GetKernelSize().cols_;

	size_t shift_idx_layer_kernel_lastlayer = 0;
	size_t shift_idx_layer_kernel_layer = 0;
	size_t idx_layer_kernel = 0;

	std::vector<float> vec_kernel((layer_kernel_rows * layer_kernel_cols), 0.0);

	for (size_t idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
	{
		for (size_t idx_lastlayer_outmap = 0; idx_lastlayer_outmap < lastlayer_outmap_num; idx_lastlayer_outmap++)
		{
			/***load kernel_ weight***/
			char str_file_kernel_1[1000];
			sprintf(str_file_kernel_1, "%s_%d_%d.txt", str_File_Kernel, idx_lastlayer_outmap, idx_layer_outmap);
			printf("%s\n", str_file_kernel_1);
			FILE* fp_kernel = fopen(str_file_kernel_1, "r");

			for (size_t mm = 0; mm < layer_kernel_rows; mm++)
			{
				for (size_t nn = 0; nn < layer_kernel_cols; nn++)
				{
					shift_idx_layer_kernel_lastlayer = idx_lastlayer_outmap * layer_outmap_num * layer_kernel_rows * layer_kernel_cols;
					shift_idx_layer_kernel_layer = idx_layer_outmap * layer_kernel_rows * layer_kernel_cols;
					idx_layer_kernel = shift_idx_layer_kernel_lastlayer + shift_idx_layer_kernel_layer + (mm * layer_kernel_cols + nn);

					fscanf(fp_kernel, "%f ", (vec_kernel.data()+(mm * layer_kernel_cols + nn)));
					r_layer.vec_kernel_.at(idx_layer_kernel) = vec_kernel.at(mm * layer_kernel_cols + nn);
					printf("kernel_: %f\n", r_layer.vec_kernel_.at(idx_layer_kernel));
				}

			}
			fclose(fp_kernel);
		}
	}
}

void CUDA_Algo_Lib::CUDACNN::Inference(CUDA_Algo_Lib::DatasetLoadingParamPKG& r_dataset_param)
{
	std::cout << "Start Inference" << std::endl;

	size_t total_false = 0, false_1 = 0, false_2 = 0, predict, real;
	size_t total_num_iter = r_dataset_param.total_num_images_ / batch_size_;

	float* p_inference_batch_data = nullptr;
	float* p_inference_batch_label = nullptr;
	std::vector<float> vec_inference_batch_data;
	std::vector<float> vec_inference_batch_label;
	vec_inference_batch_data.reserve(batch_size_ * r_dataset_param.channels_image_ * r_dataset_param.rows_image_ * r_dataset_param.cols_image_);
	vec_inference_batch_data.resize(batch_size_ * r_dataset_param.channels_image_ * r_dataset_param.rows_image_ * r_dataset_param.cols_image_);
	vec_inference_batch_label.reserve(batch_size_ * r_dataset_param.num_output_cls_);
	vec_inference_batch_label.resize(batch_size_ * r_dataset_param.num_output_cls_);

	FILE* p_file_error_predict_neg = fopen("./outputdata/error_predict_neg_filename.txt", "w");
	FILE* p_file_error_predict_pos = fopen("./outputdata/error_predict_pos_filename.txt", "w");
	for (size_t idx_iteration = 0; idx_iteration < total_num_iter; idx_iteration++)
	{
		std::cout << "NO.of iteration(testing): " << idx_iteration << std::endl;
		size_t idx_inference_dataset_batch = idx_iteration % (r_dataset_param.total_num_images_ / batch_size_);
		for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
		{
			std::cout << "NO.of batch(testing): " << idx_batch << std::endl;
		
			std::vector<float>::iterator shift_begin_iter_loaded_dataset_batch_data;
			std::vector<float>::iterator shift_begin_iter_loaded_dataset_batch_label;
			shift_begin_iter_loaded_dataset_batch_data = r_dataset_param.vec_images_.begin() + (idx_inference_dataset_batch * batch_size_ * r_dataset_param.channels_image_ * r_dataset_param.rows_image_ * r_dataset_param.cols_image_);
			shift_begin_iter_loaded_dataset_batch_label = r_dataset_param.vec_labels_.begin() + (idx_inference_dataset_batch * batch_size_ * r_dataset_param.num_output_cls_);
			vec_inference_batch_data.assign(shift_begin_iter_loaded_dataset_batch_data, (shift_begin_iter_loaded_dataset_batch_data + (batch_size_ * r_dataset_param.channels_image_ * r_dataset_param.rows_image_ * r_dataset_param.cols_image_)));
			vec_inference_batch_label.assign(shift_begin_iter_loaded_dataset_batch_label, (shift_begin_iter_loaded_dataset_batch_label + (batch_size_ * r_dataset_param.num_output_cls_)));

		}

		Forward(vec_inference_batch_data.data());
		CUDA_Algo_Lib::VECCUDACNNLayers::iterator iter = vec_layers_.end();
		iter--;
		for (size_t idx_batch = 0; idx_batch < batch_size_; idx_batch++)
		{
			std::cout << idx_batch << std::endl;

			size_t layer_outmap_num = (*iter).GetOutMapNum();
			size_t layer_outmap_rows = (*iter).GetMapSize().rows_;
			size_t layer_outmap_cols = (*iter).GetMapSize().cols_;
			size_t shift_idx_layer_batch_map = idx_batch * layer_outmap_num * layer_outmap_rows * layer_outmap_cols;
			float* p_layer_batchmap = (*iter).vec_output_maps_.data() + shift_idx_layer_batch_map;
			predict = FindIndex(p_layer_batchmap, layer_outmap_num, layer_outmap_rows, layer_outmap_cols);

			float* p_batch_gt_label = vec_inference_batch_label.data() + (idx_batch * r_dataset_param.num_output_cls_);
			real = FindIndex(p_batch_gt_label, r_dataset_param.num_output_cls_);


			//predict For batch size=2
			if (0 == idx_batch) {
				if (predict != real)
				{
					false_1++;
					//num_charaters_neg1 = sprintf(_input_negfilename, "%s%d%s", _negfilepath, idx_iteration, _imgfileextension);
					//printf("error predict-number of charaters: %d, string: \"%s\"\n", num_charaters_neg1, _input_negfilename);
					//fprintf(p_file_error_predict_neg, "%s\n", _input_negfilename);

				}
			}
			else if (1 == idx_batch) {
				if (predict != real)
				{
					false_2++;
					//num_charaters_pos1 = sprintf(_input_posfilename, "%s%d%s", _posfilepath, idx_iteration, _imgfileextension);
					//num_charaters_pos1 = sprintf(_input_posfilename, "%s%d", _posfilepath, idx_iteration);
					//printf("error predict-number of charaters: %d, string: \"%s\"\n", num_charaters_pos1, _input_posfilename);
					//fprintf(p_file_error_predict_pos, "%s\n", _input_posfilename);
				}
			}
		}
	}

	total_false = false_1 + false_2;

	std::cout << "+++++++Finish Inference+++++++" << std::endl;
	std::cout << "Error predict number of neg: " << false_1 << std::endl;
	std::cout << "Error rate of neg: " << (float)false_1 / (float)r_dataset_param.num_neg_images_ << std::endl;
	std::cout << "Error predict number of pos: " << false_2 << std::endl;
	std::cout << "Error rate of pos: " << (float)false_2 / (float)r_dataset_param.num_pos_images_ << std::endl;
	std::cout << "Error predict total number: " << total_false << std::endl;
	std::cout << "Total error rate: " << (float)total_false / (float)r_dataset_param.total_num_images_ << std::endl << std::endl;

	FILE* p_file_false_metrics;
	p_file_false_metrics = fopen("./outputdata/false_metrics.txt", "a");
	/*
	if( (err=fopen_s(&p_file_false_metrics, "fausePrun.txt", "a")) != 0 )
		exit(1) ;
	*/
	CUDA_Algo_Lib::g_idx_epoch++;
	fprintf(p_file_false_metrics, "epoch: %4d\n", CUDA_Algo_Lib::g_idx_epoch);
	fprintf(p_file_false_metrics, "neg: %4d %8f\n", false_1, (float)false_1 / (float)r_dataset_param.num_neg_images_);
	fprintf(p_file_false_metrics, "pos: %4d %8f\n", false_2, (float)false_2 / (float)r_dataset_param.num_pos_images_);
	fprintf(p_file_false_metrics, "total: %4d %8f\n\n", total_false, (float)total_false / (float)r_dataset_param.total_num_images_);
	fclose(p_file_false_metrics);
	fclose(p_file_error_predict_pos);
	fclose(p_file_error_predict_neg);

}
