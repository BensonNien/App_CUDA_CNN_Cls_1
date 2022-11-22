#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace CUDA_Algo_Lib 
{
	class RectSize
	{
	public:
		size_t rows_;
		size_t cols_;

		RectSize()
		{
			rows_ = 0;
			cols_ = 0;
		}
		~RectSize() = default;

		RectSize(size_t rows, size_t cols)
		{
			rows_ = rows;
			cols_ = cols;
		}

		RectSize Divide(RectSize scalesize)
		{
			size_t rows = rows_ / scalesize.rows_;
			size_t cols = cols_ / scalesize.cols_;
			if (rows * scalesize.rows_ != rows_ || cols * scalesize.cols_ != cols_)
			{
				std::cout << this << "can not divide" << std::endl;
			}
			return RectSize(rows, cols);
		}

		RectSize Substract(RectSize s, size_t append)
		{
			size_t rows = rows_ - s.rows_ + append;
			size_t cols = cols_ - s.cols_ + append;
			return RectSize(rows, cols);
		}

	};

	struct DatasetLoadingParamPKG {
		std::vector<float> vec_images_;
		std::vector<float> vec_labels_;
		size_t total_num_images_ = 0;
		size_t num_pos_images_ = 0;
		size_t num_neg_images_ = 0;
		size_t rows_image_ = 0;
		size_t cols_image_ = 0;
		size_t channels_image_ = 0;
		size_t num_output_cls_ = 0;
		std::string pos_images_root_path_;
		std::string neg_images_root_path_;
		std::string images_ext_;

		DatasetLoadingParamPKG(size_t num_pos_images, size_t num_neg_images,
			size_t rows_image, size_t cols_image, size_t channels_image, size_t num_output_cls,
			std::string pos_images_root_path, std::string neg_images_root_path,
			std::string images_ext)
		{

			total_num_images_ = num_pos_images + num_neg_images;
			num_pos_images_ = num_pos_images;
			num_neg_images_ = num_neg_images;
			rows_image_ = rows_image;
			cols_image_ = cols_image;
			channels_image_ = channels_image;
			num_output_cls_ = num_output_cls;

			pos_images_root_path_ = pos_images_root_path;
			neg_images_root_path_ = neg_images_root_path;
			images_ext_ = images_ext;

			vec_images_.reserve(total_num_images_ * channels_image_ * rows_image_ * cols_image_);
			vec_images_.resize(total_num_images_ * channels_image_ * rows_image_ * cols_image_);
			vec_labels_.reserve(total_num_images_ * num_output_cls_);
			vec_labels_.resize(total_num_images_ * num_output_cls_);

		}

	};
}
