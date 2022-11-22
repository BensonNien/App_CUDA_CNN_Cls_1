#pragma once
/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description:
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
#include "CPUCNNDataset.h"
#include "CPUCNNLayer.h"

namespace CPU_Algo_Lib
{
	// CNNCls
	typedef std::vector<CPU_Algo_Lib::CPUCNNLayer> VECCPUCNNLayers;

	//Builder some VECCPUCNNLayers that you want
	class CPUCNNLayerCreater
	{
	public:
		VECCPUCNNLayers vec_layers_;

		CPUCNNLayerCreater() {};
		CPUCNNLayerCreater(CPU_Algo_Lib::CPUCNNLayer layer) {
			vec_layers_.push_back(layer);
		}
		void AddLayer(CPU_Algo_Lib::CPUCNNLayer layer)
		{
			vec_layers_.push_back(layer);
		}
	};

	class CPUCNN
	{
	private:
		VECCPUCNNLayers vec_layers_;
		size_t layer_num_;
		size_t batch_size_;
		float eta_conv_;
		float alpha_conv_;
		float eta_fc_;
		float alpha_fc_;

	public:
		CPUCNN(CPUCNNLayerCreater layer_creater, size_t batch_size)
		{
			eta_conv_ = 0.006; //learning rate 
			alpha_conv_ = 0.2;//momentum rate
			eta_fc_ = 0.006; //learning rate
			alpha_fc_ = 0.2;//momentum rate
			vec_layers_ = layer_creater.vec_layers_;
			layer_num_ = vec_layers_.size();
			batch_size_ = batch_size;
			Setup(batch_size);
			SetupTest(batch_size);

		};

		~CPUCNN() = default;

		void SetBatchsize(size_t batchsize) {
			batch_size_ = batchsize;
		}
		void Train(CPU_Algo_Lib::DatasetLoadingParamPKG& r_dataset_param);
		void Inference(CPU_Algo_Lib::DatasetLoadingParamPKG& r_dataset_param);
		void Setup(size_t batch_size);// builder CPUCNN with batch_size_ and initialize some parameters of each layer
		void SetupTest(size_t batch_size);

		//back-propagation
		void BackPropagation(float* p_batch_data, float* p_batch_label);
		void SetOutLayerErrors(float* p_input_maps, float* p_target_labels);
		void SetHiddenLayerErrors();
		void SetFCHiddenLayerErrors(CPU_Algo_Lib::CPUCNNLayer& r_lastlayer, CPU_Algo_Lib::CPUCNNLayer& r_layer, CPU_Algo_Lib::CPUCNNLayer& r_nextlayer);
		void SetSampErrors(CPU_Algo_Lib::CPUCNNLayer& r_layer, CPU_Algo_Lib::CPUCNNLayer& r_nextlayer);
		void SetConvErrors(CPU_Algo_Lib::CPUCNNLayer& r_layer, CPU_Algo_Lib::CPUCNNLayer& r_nextlayer);

		void UpdateKernels(CPU_Algo_Lib::CPUCNNLayer& r_layer, CPU_Algo_Lib::CPUCNNLayer& r_lastlayer, char* str_File_Kernel, float eta, float alpha);
		void UpdateBias(CPU_Algo_Lib::CPUCNNLayer& r_layer, char* str_File_Bias, float eta);
		void UpdateParas();

		//forward
		void Forward(float* p_batch_data);
		void SetInLayerOutput(float* p_batch_data);
		void SetConvOutput(CPU_Algo_Lib::CPUCNNLayer& r_layer, CPU_Algo_Lib::CPUCNNLayer& r_lastlayer);
		void SetSampOutput(CPU_Algo_Lib::CPUCNNLayer& r_layer, CPU_Algo_Lib::CPUCNNLayer& r_lastlayer);
		void SetFCHLayerOutput(CPU_Algo_Lib::CPUCNNLayer& r_layer, CPU_Algo_Lib::CPUCNNLayer& r_lastlayer);
		void SetOutLayerOutput(CPU_Algo_Lib::CPUCNNLayer& r_layer, CPU_Algo_Lib::CPUCNNLayer& r_lastlayer);

		//load parameter
		void LoadParas();
		void LoadBias(CPU_Algo_Lib::CPUCNNLayer& r_layer, char* str_File_Bias);
		void LoadKernels(CPU_Algo_Lib::CPUCNNLayer& r_layer, CPU_Algo_Lib::CPUCNNLayer& r_lastlayer, char* str_File_Kernel);
	};
}