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
#include "CUDACNNLayer.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CUDA_Algo_Lib
{
	// CNNCls
	typedef std::vector<CUDA_Algo_Lib::CUDACNNLayer> VECCUDACNNLayers;

	//Builder some VECCUDACNNLayers that you want
	class CUDACNNLayerCreater
	{
	public:
		VECCUDACNNLayers vec_layers_;

		CUDACNNLayerCreater() = default;
		CUDACNNLayerCreater(CUDA_Algo_Lib::CUDACNNLayer layer) {
			vec_layers_.push_back(layer);
		}
		void AddLayer(CUDA_Algo_Lib::CUDACNNLayer layer)
		{
			vec_layers_.push_back(layer);
		}
	};

	class CUDACNN
	{
	private:
		VECCUDACNNLayers vec_layers_;
		size_t layer_num_;
		size_t batch_size_;
		float eta_conv_;
		float alpha_conv_;
		float eta_fc_;
		float alpha_fc_;

	public:
		CUDACNN(CUDACNNLayerCreater layer_creater, size_t batch_size)
		{
			eta_conv_ = 0.0018; //learning rate 
			alpha_conv_ = 0.01;//momentum rate
			eta_fc_ = 0.0018; //learning rate
			alpha_fc_ = 0.01;//momentum rate
			vec_layers_ = layer_creater.vec_layers_;
			layer_num_ = vec_layers_.size();
			batch_size_ = batch_size;
			//InitCUDADevice();
			Setup(batch_size);
			SetupTest(batch_size);

		};

		~CUDACNN() = default;

		cudaError_t static InitCUDADevice();

		void SetBatchsize(size_t batchsize) {
			batch_size_ = batchsize;
		}
		void Train(CUDA_Algo_Lib::DatasetLoadingParamPKG& r_dataset_param);
		void Inference(CUDA_Algo_Lib::DatasetLoadingParamPKG& r_dataset_param);
		void Setup(size_t batch_size);// builder CUDACNN with batch_size_ and initialize some parameters of each layer
		void SetupTest(size_t batch_size);

		//back-propagation
		void BackPropagation(float* p_batch_data, float* p_batch_label);
		void SetOutLayerErrors(float* p_input_maps, float* p_target_labels);
		void SetHiddenLayerErrors();		
		void SetSampErrors(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_nextlayer);
		void SetConvErrors(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_nextlayer);
		cudaError_t SetFCHiddenLayerErrors(CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer, CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_nextlayer);

		void UpdateKernels(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer, char* str_File_Kernel, float eta, float alpha);
		void UpdateBias(CUDA_Algo_Lib::CUDACNNLayer& r_layer, char* str_File_Bias, float eta);
		void UpdateParas();

		//forward
		void Forward(float* p_batch_data);
		cudaError_t SetInLayerOutput(float* p_batch_data);
		cudaError_t SetConvOutput(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer);
		cudaError_t SetSampOutput(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer);
		cudaError_t SetFCHLayerOutput(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer);
		cudaError_t SetOutLayerOutput(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer);

		//load parameter
		void LoadParas();
		void LoadBias(CUDA_Algo_Lib::CUDACNNLayer& r_layer, char* str_File_Bias);
		void LoadKernels(CUDA_Algo_Lib::CUDACNNLayer& r_layer, CUDA_Algo_Lib::CUDACNNLayer& r_lastlayer, char* str_File_Kernel);
	};
}