#pragma once
#include <fstream> 
#include "opencv2/opencv.hpp"
#include "CUDACNNDataStruct.cuh"

namespace CUDA_Algo_Lib
{
	class CNNDataset {
	public:
		// Big5 Funcs
		CNNDataset() = default;
		~CNNDataset() = default;

		// Static Member Funcs
		static void Load(CUDA_Algo_Lib::DatasetLoadingParamPKG& r_dataset_param);
	};
}
