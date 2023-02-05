#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "CPUCNNDataStruct.h"
#include "CPUCNNDataset.h"
#include "CPUCNNCls.h"
#include "CUDACNNDataStruct.cuh"
#include "CUDACNNDataset.cuh"
#include "CUDACNNCls.cuh"

void RunCPUAIFlow();
cudaError_t RunCUDAAIFlow();

int main()
{
    std::cout << "\n====================== Prog. Start ======================\n";
    
    //RunCPUAIFlow();
    RunCUDAAIFlow();

    std::cout << "\n====================== Prog. End ======================\n";
    return 0;
}

void RunCPUAIFlow()
{
    std::cout << "\n============ Run CPU AI Flow Start ============\n";
    // initialize input data

    size_t num_pos_train_images = 1000;
    size_t num_neg_train_images = 1000;
    size_t num_train_images = num_pos_train_images + num_neg_train_images;
    size_t num_pos_validation_images = 1000;
    size_t num_neg_validation_images = 1000;
    size_t num_validation_images = num_pos_validation_images + num_neg_validation_images;
    size_t num_pos_test_images = 1000;
    size_t num_neg_test_images = 1000;
    size_t num_test_images = num_pos_test_images + num_neg_test_images;
    size_t rows_image = 64;
    size_t cols_image = 32;
    size_t channels_image = 3;
    size_t num_output_cls = 2;

    std::string pos_train_images_root_path = ".\\Pedestrian_TrainingDataset_PNG\\64x32_part_balance_v2\\pos\\Training_p_";
    std::string neg_train_images_root_path = ".\\Pedestrian_TrainingDataset_PNG\\64x32_part_balance_v2\\neg\\Training_n_";
    std::string pos_validation_images_root_path = pos_train_images_root_path;
    std::string neg_validation_images_root_path = neg_train_images_root_path;
    std::string pos_test_images_root_path = pos_train_images_root_path;
    std::string neg_test_images_root_path = neg_train_images_root_path;
    std::string images_ext = ".png";


    CPU_Algo_Lib::DatasetLoadingParamPKG train_dataset_param(num_pos_train_images, num_neg_train_images,
        rows_image, cols_image, channels_image, num_output_cls,
        pos_train_images_root_path, neg_train_images_root_path,
        images_ext);

    CPU_Algo_Lib::DatasetLoadingParamPKG validation_dataset_param(num_pos_train_images, num_neg_train_images,
        rows_image, cols_image, channels_image, num_output_cls,
        pos_train_images_root_path, neg_train_images_root_path,
        images_ext);

    CPU_Algo_Lib::DatasetLoadingParamPKG test_dataset_param(num_pos_train_images, num_neg_train_images,
        rows_image, cols_image, channels_image, num_output_cls,
        pos_train_images_root_path, neg_train_images_root_path,
        images_ext);

    CPU_Algo_Lib::CNNDataset::Load(train_dataset_param);
    CPU_Algo_Lib::CNNDataset::Load(validation_dataset_param);
    CPU_Algo_Lib::CNNDataset::Load(test_dataset_param);

    // constructor CPU_Algo_Lib::CPUCNN
    CPU_Algo_Lib::CPUCNNLayerCreater layer_creater;
    CPU_Algo_Lib::CPUCNNLayer layer;
    layer_creater.AddLayer(layer.CreateInputLayer(3, CPU_Algo_Lib::RectSize(rows_image, cols_image)));//image's channel, image CPU_Algo_Lib::RectSize

    layer_creater.AddLayer(layer.CreateConvLayer(3, 6, CPU_Algo_Lib::RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel CPU_Algo_Lib::RectSize
    layer_creater.AddLayer(layer.CreateSampLayer(CPU_Algo_Lib::RectSize(2, 2)));//Downsampling layer kernel CPU_Algo_Lib::RectSize
    layer_creater.AddLayer(layer.CreateConvLayer(6, 12, CPU_Algo_Lib::RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel CPU_Algo_Lib::RectSize
    layer_creater.AddLayer(layer.CreateSampLayer(CPU_Algo_Lib::RectSize(2, 2)));//Downsampling layer kernel CPU_Algo_Lib::RectSize
    layer_creater.AddLayer(layer.CreateConvLayer(12, 20, CPU_Algo_Lib::RectSize(4, 4)));//convolutional layer output feature map's deep number, kernel CPU_Algo_Lib::RectSize
    layer_creater.AddLayer(layer.CreateSampLayer(CPU_Algo_Lib::RectSize(2, 2)));//Downsampling layer kernel CPU_Algo_Lib::RectSize

    layer_creater.AddLayer(layer.CreateFullyConnectedHiddenLayer(20, 14, num_output_cls));//Fully connected hidden layer node number
    layer_creater.AddLayer(layer.CreateOutputLayer(num_output_cls));//output layer node number

    CPU_Algo_Lib::CPUCNN cnn = CPU_Algo_Lib::CPUCNN(layer_creater, 2);// batchsize

    float t0 = CPU_Algo_Lib::EvlElapsedTime();
    //cnn.LoadParas();//load kernel weight & bias

    for (size_t i = 0; i < 50; i++)//i is training epoch
    {
        std::cout << "No.of Training: " << i << std::endl;
        float t1 = CPU_Algo_Lib::EvlElapsedTime();
        cnn.Train(train_dataset_param);
        float t2 = CPU_Algo_Lib::EvlElapsedTime();
        std::cout << t2 - t1 << " s" << std::endl << i + 1 << std::endl;
        std::cout << "No.of Testing: " << i << std::endl;
        cnn.Inference(validation_dataset_param);
    }

    float te = CPU_Algo_Lib::EvlElapsedTime();
    std::cout << "total: " << te - t0 << " s" << std::endl;

    //for testing
    cnn.LoadParas();//load kernel weight & bias
    cnn.Inference(test_dataset_param);

    std::cout << "\n============ Run CPU AI Flow End ============\n";
}

cudaError_t RunCUDAAIFlow()
{
    std::cout << "\n============ Run CUDA AI Flow Start ============\n";
    
    cudaError_t cudaStatus;

    // initialize input data
    size_t num_pos_train_images = 1000;
    size_t num_neg_train_images = 1000;
    size_t num_train_images = num_pos_train_images + num_neg_train_images;
    size_t num_pos_validation_images = 1000;
    size_t num_neg_validation_images = 1000;
    size_t num_validation_images = num_pos_validation_images + num_neg_validation_images;
    size_t num_pos_test_images = 1000;
    size_t num_neg_test_images = 1000;
    size_t num_test_images = num_pos_test_images + num_neg_test_images;
    size_t rows_image = 64;
    size_t cols_image = 32;
    size_t channels_image = 3;
    size_t num_output_cls = 2;

    std::string pos_train_images_root_path = ".\\Pedestrian_TrainingDataset_PNG\\64x32_part_balance_v2\\pos\\Training_p_";
    std::string neg_train_images_root_path = ".\\Pedestrian_TrainingDataset_PNG\\64x32_part_balance_v2\\neg\\Training_n_";
    std::string pos_validation_images_root_path = pos_train_images_root_path;
    std::string neg_validation_images_root_path = neg_train_images_root_path;
    std::string pos_test_images_root_path = pos_train_images_root_path;
    std::string neg_test_images_root_path = neg_train_images_root_path;
    std::string images_ext = ".png";


    CUDA_Algo_Lib::DatasetLoadingParamPKG train_dataset_param(num_pos_train_images, num_neg_train_images,
        rows_image, cols_image, channels_image, num_output_cls,
        pos_train_images_root_path, neg_train_images_root_path,
        images_ext);

    CUDA_Algo_Lib::DatasetLoadingParamPKG validation_dataset_param(num_pos_train_images, num_neg_train_images,
        rows_image, cols_image, channels_image, num_output_cls,
        pos_train_images_root_path, neg_train_images_root_path,
        images_ext);

    CUDA_Algo_Lib::DatasetLoadingParamPKG test_dataset_param(num_pos_train_images, num_neg_train_images,
        rows_image, cols_image, channels_image, num_output_cls,
        pos_train_images_root_path, neg_train_images_root_path,
        images_ext);

    CUDA_Algo_Lib::CNNDataset::Load(train_dataset_param);
    CUDA_Algo_Lib::CNNDataset::Load(validation_dataset_param);
    CUDA_Algo_Lib::CNNDataset::Load(test_dataset_param);

    // constructor CUDA_Algo_Lib::CUDACNN
    CUDA_Algo_Lib::CUDACNN::InitCUDADevice();
    //system("pause");

    CUDA_Algo_Lib::CUDACNNLayerCreater layer_creater;
    CUDA_Algo_Lib::CUDACNNLayer layer;
    layer_creater.AddLayer(layer.CreateInputLayer(3, CUDA_Algo_Lib::RectSize(rows_image, cols_image)));//image's channel, image CUDA_Algo_Lib::RectSize

    layer_creater.AddLayer(layer.CreateConvLayer(3, 6, CUDA_Algo_Lib::RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel CUDA_Algo_Lib::RectSize
    layer_creater.AddLayer(layer.CreateSampLayer(CUDA_Algo_Lib::RectSize(2, 2)));//Downsampling layer kernel CUDA_Algo_Lib::RectSize
    layer_creater.AddLayer(layer.CreateConvLayer(6, 12, CUDA_Algo_Lib::RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel CUDA_Algo_Lib::RectSize
    layer_creater.AddLayer(layer.CreateSampLayer(CUDA_Algo_Lib::RectSize(2, 2)));//Downsampling layer kernel CUDA_Algo_Lib::RectSize
    layer_creater.AddLayer(layer.CreateConvLayer(12, 20, CUDA_Algo_Lib::RectSize(4, 4)));//convolutional layer output feature map's deep number, kernel CUDA_Algo_Lib::RectSize
    layer_creater.AddLayer(layer.CreateSampLayer(CUDA_Algo_Lib::RectSize(2, 2)));//Downsampling layer kernel CUDA_Algo_Lib::RectSize

    layer_creater.AddLayer(layer.CreateFullyConnectedHiddenLayer(20, 14, num_output_cls));//Fully connected hidden layer node number
    layer_creater.AddLayer(layer.CreateOutputLayer(num_output_cls));//output layer node number

    CUDA_Algo_Lib::CUDACNN cnn = CUDA_Algo_Lib::CUDACNN(layer_creater, 2);// batchsize

    float t0 = CUDA_Algo_Lib::EvlElapsedTime();
    // cnn.LoadParas();//load kernel weight & bias

    for (size_t i = 0; i < 2; i++)//i is training epoch
    {
        std::cout << "No.of Training: " << i << std::endl;
        float t1 = CUDA_Algo_Lib::EvlElapsedTime();
        cnn.Train(train_dataset_param);
        float t2 = CUDA_Algo_Lib::EvlElapsedTime();
        std::cout << t2 - t1 << " s" << std::endl << i + 1 << std::endl;
        std::cout << "No.of Testing: " << i << std::endl;
        cnn.Inference(validation_dataset_param);
        //system("pause");
    }

    float te = CUDA_Algo_Lib::EvlElapsedTime();
    std::cout << "total: " << te - t0 << " s" << std::endl;

    //for testing
    cnn.LoadParas();//load kernel weight & bias
    cnn.Inference(test_dataset_param);

    std::cout << "\n============ Run CUDA AI Flow End ============\n";
    return cudaStatus;
}

