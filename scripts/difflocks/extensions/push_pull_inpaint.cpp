#include <torch/extension.h>

#include <iostream>

// #include "cuda/SplatSliceGPU.cuh"


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x, d) TORCH_CHECK((x.dim() == (d)), #x " must be a tensor with ", d, " dimensions, but has shape ", x.sizes())
#define CHECK_SIZE(x, d, s) TORCH_CHECK((x.size(d) == (s)), #x " must have ", s, " entries at dimension ", d, ", but has ", x.size(d), " entries")

#define MAX_CHANNELS 128

// CUDA forward declarations
std::tuple<torch::Tensor, torch::Tensor>
		push_pull_inpaint_recursion_cuda(
			const torch::Tensor& mask,
			const torch::Tensor& data);




torch::Tensor push_pull_inpaint(
	const torch::Tensor& mask,
	const torch::Tensor& data)
{
	//check input
	CHECK_CUDA(mask);
	CHECK_CONTIGUOUS(mask);
	CHECK_DIM(mask, 3);
	int64_t B = mask.size(0);
	int64_t H = mask.size(1);
	int64_t W = mask.size(2);

	CHECK_CUDA(data);
	CHECK_CONTIGUOUS(data);
	CHECK_DIM(data, 4);
	CHECK_SIZE(data, 0, B);
	int64_t C = data.size(1);
	CHECK_SIZE(data, 2, H);
	CHECK_SIZE(data, 3, W);
	TORCH_CHECK(C < MAX_CHANNELS, "Inpainting::fastInpaint only supports up to 128 channels, but got " + std::to_string(C));

	//inpaint recursivly
	return std::get<1>(push_pull_inpaint_recursion_cuda(mask, data));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("push_pull_inpaint", &push_pull_inpaint);
}