#pragma once
/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description: GPU (CUDA) ver.
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
#include "GPUCNNLayer.cuh"

// CNNCls
typedef std::vector<GPUCNNLayer> VECGPUCNNLayers;

//Builder some VECGPUCNNLayers that you want
class GPUCNNLayerCreater
{
public:
	VECGPUCNNLayers vec_layers_;

	GPUCNNLayerCreater() = default;
	GPUCNNLayerCreater(GPUCNNLayer layer) {
		vec_layers_.push_back(layer);
	}
	void AddLayer(GPUCNNLayer layer)
	{
		vec_layers_.push_back(layer);
	}
};

class GPUCNN
{
private:
	VECGPUCNNLayers vec_layers_;
	size_t layer_num_;
	size_t batch_size_;
	float eta_conv_;
	float alpha_conv_;
	float eta_fc_;
	float alpha_fc_;

public:
	GPUCNN(GPUCNNLayerCreater layer_creater, size_t batch_size)
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
	~GPUCNN() {};

	void SetBatchsize(size_t batchsize) {
		batch_size_ = batchsize;
	}
	
	void Train(GPUCNNDataStruct_Lib::DatasetLoadingStructParamPKG& r_dataset_param);
	void Inference(GPUCNNDataStruct_Lib::DatasetLoadingStructParamPKG& r_dataset_param);
	void Setup(size_t batch_size);// builder GPUCNN with batch_size_ and initialize some parameters of each layer
	void SetupTest(size_t batch_size);

	//back-propagation
	void BackPropagation(float* p_batch_data, float* p_batch_label);
	void SetOutLayerErrors(float* p_input_maps, float* p_target_labels);
	void SetHiddenLayerErrors();
	void SetFCHiddenLayerErrors(GPUCNNLayer& Lastlayer, GPUCNNLayer& layer, GPUCNNLayer& nextLayer);
	void SetSampErrors(GPUCNNLayer& layer, GPUCNNLayer& nextLayer);
	void SetConvErrors(GPUCNNLayer& layer, GPUCNNLayer& nextLayer);

	void UpdateKernels(GPUCNNLayer& layer, GPUCNNLayer& lastLayer, char* str_File_Kernel, float eta, float alpha);
	void UpdateBias(GPUCNNLayer& layer, char* str_File_Bias, float eta);
	void UpdateParas();

	//forward
	void Forward(float* p_batch_data);
	void SetInLayerOutput(float* p_batch_data);
	void SetConvOutput(GPUCNNLayer& layer, GPUCNNLayer& lastLayer);
	void SetSampOutput(GPUCNNLayer& layer, GPUCNNLayer& lastLayer);
	void SetFCHLayerOutput(GPUCNNLayer& layer, GPUCNNLayer& lastLayer);
	void SetOutLayerOutput(GPUCNNLayer& layer, GPUCNNLayer& lastLayer);

	//load parameter
	void LoadParas();
	void LoadBias(GPUCNNLayer& layer, char* str_File_Bias);
	void LoadKernels(GPUCNNLayer& layer, GPUCNNLayer& lastLayer, char* str_File_Kernel);
};
