// std::tuple<torch::Tensor, torch::Tensor>
// 		fastInpaint_recursion(
// 			const torch::Tensor& mask,
// 			const torch::Tensor& data)
// 	{
// 		int64_t B = data.size(0);
// 		int64_t C = data.size(1);
// 		int64_t H = data.size(2);
// 		int64_t W = data.size(3);

// 		if (H <= 1 && W <= 1)
// 			return std::make_tuple(mask, data); //end of recursion

// 		int64_t oH = H / 2;
// 		int64_t oW = W / 2;

// 		//prepare launching
// 		cuMat::Context& ctx = cuMat::Context::current();
// 		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

// 		//downsample
// 		torch::Tensor maskLow = torch::empty({ B, oH, oW }, mask.options());
// 		torch::Tensor dataLow = torch::empty({ B, C, oH, oW }, data.options());
// 		AT_DISPATCH_FLOATING_TYPES(data.type(), "FastInpaintingKernel_Down", ([&]
// 		{
// 			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
// 				oW, oH, B, FastInpaintingKernel_Down<scalar_t>);
// 			FastInpaintingKernel_Down<scalar_t>
// 				<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >
// 				(cfg.virtual_size,
// 					mask.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
// 					data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
// 					maskLow.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
// 					dataLow.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
// 		}));
// 		CUMAT_CHECK_ERROR();

// 		//recursion
// 		const auto tuple = fastInpaint_recursion(maskLow, dataLow);
// 		const auto& maskLow2 = std::get<0>(tuple);
// 		const auto& dataLow2 = std::get<1>(tuple);

// 		//upsample
// 		torch::Tensor maskHigh = torch::empty({ B, H, W }, mask.options());
// 		torch::Tensor dataHigh = torch::empty({ B, C, H, W }, data.options());
// 		AT_DISPATCH_FLOATING_TYPES(data.type(), "FastInpaintingKernel_Up", ([&]
// 		{
// 			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
// 				W, H, B, FastInpaintingKernel_Up<scalar_t>);
// 			FastInpaintingKernel_Up<scalar_t>
// 				<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >
// 				(cfg.virtual_size,
// 					mask.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
// 					data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
// 					maskLow2.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
// 					dataLow2.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
// 					maskHigh.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
// 					dataHigh.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
// 		}));
// 		CUMAT_CHECK_ERROR();

// 		//done
// 		return std::make_tuple(maskHigh, dataHigh);
// 	}



#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h" //needed for threadIdx and blockDim

#include <cuMat/src/Context.h>
// #include <cuMat/Core>
#include <ATen/cuda/CUDAContext.h>
#include <stack>

// #ifdef RENDERER_HAS_INPAINTING

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x, d) TORCH_CHECK((x.dim() == (d)), #x " must be a tensor with ", d, " dimensions, but has shape ", x.sizes())
#define CHECK_SIZE(x, d, s) TORCH_CHECK((x.size(d) == (s)), #x " must have ", s, " entries at dimension ", d, ", but has ", x.size(d), " entries")

#define MAX_CHANNELS 128













// void splat_texture(float* texture, const float* values, const float* uv, const int nr_values, const int val_dim, const int texture_height, const int texture_width){

//     dim3 blocks((nr_values - 1) / BLOCK_SIZE + 1, 1, 1);
//     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     CUresult res= m_program.kernel("splat_texture")
//                 .instantiate(val_dim, texture_height, texture_width)
//                 .configure(blocks, blockSize)
//                 .launch(nr_values, texture, values, uv );

// }


//         void slice_texture(float* values_not_normalized_tensor, const float* texture, const float* uv, const int nr_values, const int nr_channels_texture, const int texture_height, const int texture_width){

//             dim3 blocks((nr_values - 1) / BLOCK_SIZE + 1, 1, 1);
//             dim3 blockSize(BLOCK_SIZE, 1, 1);
//             CUresult res= m_program.kernel("slice_texture")
//                         .instantiate(nr_channels_texture, texture_height, texture_width)
//                         .configure(blocks, blockSize)
//                         .launch(nr_values, values_not_normalized_tensor, texture, uv );
//             CUDA_CHECK_CURESULT(res);
//             CUDA_CHECK_ERROR();

//         }



//         void splat_texture_backward(float* grad_values, float* grad_uv, const float* grad_texture, const float* values, const float* uv, const int nr_values, const int val_dim, const int texture_height, const int texture_width){

//             dim3 blocks((nr_values - 1) / BLOCK_SIZE + 1, 1, 1);
//             dim3 blockSize(BLOCK_SIZE, 1, 1);
//             CUresult res= m_program.kernel("splat_texture_backward")
//                         .instantiate( val_dim, texture_height, texture_width)
//                         .configure(blocks, blockSize)
//                         .launch( nr_values, grad_values, grad_uv, grad_texture, values, uv );
//             CUDA_CHECK_CURESULT(res);
//             CUDA_CHECK_ERROR();

//         }

//         //  m_impl->slice_texture_backward( grad_texture.data_ptr<float>(), grad_uv.data_ptr<float>(). //output
//         //                             grad_values.data_ptr<float>(), texture.data_ptr<float>(), uv_tensor.data_ptr<float>(), //input
//         //                             nr_values, nr_channels_texture, texture_size); //constant

//         void slice_texture_backward(float* grad_texture, float* grad_uv, const float* grad_values_not_normalized, const float* texture, const float* uv, const int nr_values, const int nr_channels_texture, const int texture_height, const int texture_width){

//             dim3 blocks((nr_values - 1) / BLOCK_SIZE + 1, 1, 1);
//             dim3 blockSize(BLOCK_SIZE, 1, 1);
//             CUresult res= m_program.kernel("slice_texture_backward")
//                         .instantiate( nr_channels_texture, texture_height, texture_width)
//                         .configure(blocks, blockSize)
//                         .launch( nr_values, grad_texture, grad_uv, grad_values_not_normalized, texture, uv );
//             CUDA_CHECK_CURESULT(res);
//             CUDA_CHECK_ERROR();

//         }






// template<int nr_channels_texture, int texture_height, int texture_width>
// __global__ void
// __launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
// slice_texture( int nr_values, float* values_not_normalized, const float* texture, const float* uv ){

//     int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

//     if(idx>=nr_values){ //don't go out of bounds
//         return;
//     }

//     //possibly needed variables
//     // int val_dim = nr_channels_texture-1; //in the texture we also store a homogeneous coords so the val dim will be -1


//     //grab pointer to the current values we are procesing
//     float* cur_val_not_normalized = values_not_normalized+idx*nr_channels_texture;
//     const float* cur_uv = uv+idx*2;

//     //the uvs are supposed to be in range [-1, 1], now we get them in range [0, texture_size-1]
//     // float u_range01 = (cur_uv[0]+1)*0.5;
//     // float v_range01 = (cur_uv[1]+1)*0.5;
//     //shift the u and v by half a pixel so that we hit the center of the pixel.
//     // scale_uv_to_hit_pixel_centers(u_range01, v_range01,  texture_width, texture_height, u_range01, v_range01);
//     // float ix=u_range01*(texture_width-1);
//     // float iy=v_range01*(texture_height-1);

//     //again taking inspiration from https://github.com/pytorch/pytorch/blob/5fc391a6465d1f7197008085bb521b6dfaccc8b3/aten/src/ATen/native/cuda/GridSampler.cuh#L28
//     float ix=((cur_uv[0] + 1.f) * texture_width - 1) / 2;
//     float iy=((cur_uv[1] + 1.f) * texture_height - 1) / 2;

//     //using align corners=true, so apparently the incorrect way of doing it
//     // float ix=((cur_uv[0] + 1.f) / 2) * (texture_width - 1);
//     // float iy=((cur_uv[1] + 1.f) / 2) * (texture_height - 1);



//     //get the coordiantes of the neighbouring 4 pixels according to the wikipedia convention  https://en.wikipedia.org/wiki/Bilinear_interpolation
//     //get the coordiantes of the neighbouring 4 pixels according to the convention of https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c
//     int ix_nw = floor(ix);
//     int iy_nw = floor(iy);
//     int ix_ne = ix_nw + 1;
//     int iy_ne = iy_nw;
//     int ix_sw = ix_nw;
//     int iy_sw = iy_nw + 1;
//     int ix_se = ix_nw + 1;
//     int iy_se = iy_nw + 1;

//     //we calculate the weights here so as to also splat even though we are a bit out of border
//     float nw = (ix_se - ix)    * (iy_se - iy);
//     float ne = (ix    - ix_sw) * (iy_sw - iy);
//     float sw = (ix_ne - ix)    * (iy    - iy_ne);
//     float se = (ix - ix_nw) * (iy - iy_nw);


//     //coordinates and if the coordinate was clipped, also set the weight to zero
//     CLIP_COORDINATES_AND_WEIGHTS(ix_nw, nw, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_nw, nw, texture_height);
//     CLIP_COORDINATES_AND_WEIGHTS(ix_ne, ne, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_ne, ne, texture_height);
//     CLIP_COORDINATES_AND_WEIGHTS(ix_sw, sw, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_sw, sw, texture_height);
//     CLIP_COORDINATES_AND_WEIGHTS(ix_se, se, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_se, se, texture_height);


//     //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
//     const float* nw_val = texture + ix_nw*nr_channels_texture + iy_nw*texture_width*nr_channels_texture;
//     const float* ne_val = texture + ix_ne*nr_channels_texture + iy_ne*texture_width*nr_channels_texture;
//     const float* sw_val = texture + ix_sw*nr_channels_texture + iy_sw*texture_width*nr_channels_texture;
//     const float* se_val = texture + ix_se*nr_channels_texture + iy_se*texture_width*nr_channels_texture;

//     //get the weigthings of the pixels
//     // float denom = 1.0/( (x2 - x1)*(y2-y1)  + 1e-7 ); //the denominator is mostly just to normalize for the size of the pixel but we can assume a pixel of size 1 and just drop this whole term
//     // float wq11=(x2-x)*(y2-y);
//     // float wq21=(x-x1)*(y2-y);
//     // float wq12=(x2-x)*(y-y1);
//     // float wq22=(x-x1)*(y-y1);


//     //retreive the value weighted between the 4 pixels
//     // float val_not_normalized[nr_channels_texture]{0};
//     for(int i=0; i<nr_channels_texture; i++){

//         cur_val_not_normalized[i]+=nw_val[i]*nw;
//         cur_val_not_normalized[i]+=ne_val[i]*ne;
//         cur_val_not_normalized[i]+=sw_val[i]*sw;
//         cur_val_not_normalized[i]+=se_val[i]*se;
//     }


// }


// template< int val_dim, int texture_height, int texture_width>
// __global__ void
// __launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
// splat_texture_backward( int nr_values, float* grad_values, float* grad_uv, const float* grad_texture, const float* values, const float* uv ){

//     int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

//     if(idx>=nr_values){ //don't go out of bounds
//         return;
//     }

//     //possibly needed variables
//     // int nr_channels_texture=val_dim+1; // in the texture we store also a homogeneous coordinate, so we have a +1
//     int nr_channels_texture=val_dim; // we assume we already have a homogneous corod

//     //grab pointer to the current values we are procesing
//     const float* cur_val = values+idx*val_dim;
//     const float* cur_uv = uv+idx*2;
//     float* cur_grad_uv = grad_uv + idx*2;
//     float* cur_grad_values = grad_values + idx*val_dim;




//     //the uvs are supposed to be in range [-1, 1], now we get them in range [0, texture_size-1]
//     // float u_range01 = (cur_uv[0]+1)*0.5;
//     // float v_range01 = (cur_uv[1]+1)*0.5;
//     //shift the u and v by half a pixel so that we hit the center of the pixel.
//     // scale_uv_to_hit_pixel_centers(u_range01, v_range01,  texture_width, texture_height, u_range01, v_range01);
//     // float ix=u_range01*(texture_width-1);
//     // float iy=v_range01*(texture_height-1);

//     //again taking inspiration from https://github.com/pytorch/pytorch/blob/5fc391a6465d1f7197008085bb521b6dfaccc8b3/aten/src/ATen/native/cuda/GridSampler.cuh#L28
//     float ix=((cur_uv[0] + 1.f) * texture_width - 1) / 2;
//     float iy=((cur_uv[1] + 1.f) * texture_height - 1) / 2;

//     //using align corners=true, so apparently the incorrect way of doing it
//     // float ix=((cur_uv[0] + 1.f) / 2) * (texture_width - 1);
//     // float iy=((cur_uv[1] + 1.f) / 2) * (texture_height - 1);


//     //get the coordiantes of the neighbouring 4 pixels according to the wikipedia convention  https://en.wikipedia.org/wiki/Bilinear_interpolation
//     //get the coordiantes of the neighbouring 4 pixels according to the convention of https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c
//     int ix_nw = floor(ix);
//     int iy_nw = floor(iy);
//     int ix_ne = ix_nw + 1;
//     int iy_ne = iy_nw;
//     int ix_sw = ix_nw;
//     int iy_sw = iy_nw + 1;
//     int ix_se = ix_nw + 1;
//     int iy_se = iy_nw + 1;

//     //we calculate the weights here so as to also splat even though we are a bit out of border
//     float nw = (ix_se - ix)    * (iy_se - iy);
//     float ne = (ix    - ix_sw) * (iy_sw - iy);
//     float sw = (ix_ne - ix)    * (iy    - iy_ne);
//     float se = (ix - ix_nw) * (iy - iy_nw);


//     //coordinates and if the coordinate was clipped, also set the weight to zero
//     CLIP_COORDINATES_AND_WEIGHTS(ix_nw, nw, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_nw, nw, texture_height);
//     CLIP_COORDINATES_AND_WEIGHTS(ix_ne, ne, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_ne, ne, texture_height);
//     CLIP_COORDINATES_AND_WEIGHTS(ix_sw, sw, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_sw, sw, texture_height);
//     CLIP_COORDINATES_AND_WEIGHTS(ix_se, se, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_se, se, texture_height);



//     //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation


//     const float* d_nw_val = grad_texture + ix_nw*nr_channels_texture + iy_nw*texture_width*nr_channels_texture;
//     const float* d_ne_val = grad_texture + ix_ne*nr_channels_texture + iy_ne*texture_width*nr_channels_texture;
//     const float* d_sw_val = grad_texture + ix_sw*nr_channels_texture + iy_sw*texture_width*nr_channels_texture;
//     const float* d_se_val = grad_texture + ix_se*nr_channels_texture + iy_se*texture_width*nr_channels_texture;

//     //get the weigthings of the pixels
//     // float denom = 1.0/( (x2 - x1)*(y2-y1)  + 1e-7 ); //the denominator is mostly just to normalize for the size of the pixel but we can assume a pixel of size 1 and just drop this whole term
//     // float wq11=(x2-x)*(y2-y);
//     // float wq21=(x-x1)*(y2-y);
//     // float wq12=(x2-x)*(y-y1);
//     // float wq22=(x-x1)*(y-y1);






//     ///GRAD UV
//     float grad_u=0;
//     float grad_v=0;
//     for(int i=0; i<val_dim; i++){
//         grad_u+=cur_val[i]*(-1)*(iy_se - iy)*d_nw_val[i];
//         grad_u+=cur_val[i]*(iy_sw - iy)*d_ne_val[i];
//         grad_u+=cur_val[i]*(-1)*(iy    - iy_ne)*d_sw_val[i];
//         grad_u+=cur_val[i]*(iy - iy_nw)*d_se_val[i];

//         grad_v+=cur_val[i]*(-1)*(ix_se - ix)*d_nw_val[i];
//         grad_v+=cur_val[i]*(-1)*(ix    - ix_sw)*d_ne_val[i];
//         grad_v+=cur_val[i]*(ix_ne - ix)*d_sw_val[i];
//         grad_v+=cur_val[i]*(ix - ix_nw)*d_se_val[i];
//     }
//     // //TODO gradients with respect to the homogeneous coord is as if we splatted a value of 1
//     // grad_u+=1*(-1)*(iy_se - iy)*d_nw_val[val_dim];
//     // grad_u+=1*(iy_sw - iy)*d_ne_val[val_dim];
//     // grad_u+=1*(-1)*(iy    - iy_ne)*d_sw_val[val_dim];
//     // grad_u+=1*(iy - iy_nw)*d_se_val[val_dim];

//     // grad_v+=1*(-1)*(ix_se - ix)*d_nw_val[val_dim];
//     // grad_v+=1*(-1)*(ix    - ix_sw)*d_ne_val[val_dim];
//     // grad_v+=1*(ix_ne - ix)*d_sw_val[val_dim];
//     // grad_v+=1*(ix - ix_nw)*d_se_val[val_dim];

//     //unnormalize the grad_uv back to the [-1.1] constraint
//     // https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c
//     grad_u = grad_u * (texture_width - 1) *0.5;
//     grad_v = grad_v * (texture_height - 1) *0.5;

//     //put them in the tensor
//     cur_grad_uv[0]=grad_u;
//     cur_grad_uv[1]=grad_v;



//     //GRAD VALUES
//     for(int i=0; i<val_dim; i++){
//         cur_grad_values[i]+=d_nw_val[i]*nw;
//         cur_grad_values[i]+=d_ne_val[i]*ne;
//         cur_grad_values[i]+=d_sw_val[i]*sw;
//         cur_grad_values[i]+=d_se_val[i]*se;
//     }




// }



// template< int nr_channels_texture, int texture_height, int texture_width>
// __global__ void
// __launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
// slice_texture_backward(int nr_values, float* grad_texture, float* grad_uv, const float* grad_values_not_normalized, const float* texture, const float* uv ){


//     int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

//     if(idx>=nr_values){ //don't go out of bounds
//         return;
//     }

//     //possibly needed variables
//     // int val_dim=nr_channels_texture-1; // in the texture we store also a homogeneous coordinate, so we have a -1

//     //grab pointer to the current values we are procesing
//     const float* cur_grad_val = grad_values_not_normalized+idx*nr_channels_texture;
//     const float* cur_uv = uv+idx*2;
//     float* cur_grad_uv = grad_uv + idx*2;


//     //the uvs are supposed to be in range [-1, 1], now we get them in range [0, texture_size-1]
//     // float u_range01 = (cur_uv[0]+1)*0.5;
//     // float v_range01 = (cur_uv[1]+1)*0.5;
//     //shift the u and v by half a pixel so that we hit the center of the pixel.
//     // scale_uv_to_hit_pixel_centers(u_range01, v_range01,  texture_width, texture_height, u_range01, v_range01);
//     // float ix=u_range01*(texture_width-1);
//     // float iy=v_range01*(texture_height-1);

//     //again taking inspiration from https://github.com/pytorch/pytorch/blob/5fc391a6465d1f7197008085bb521b6dfaccc8b3/aten/src/ATen/native/cuda/GridSampler.cuh#L28
//     float ix=((cur_uv[0] + 1.f) * texture_width - 1) / 2;
//     float iy=((cur_uv[1] + 1.f) * texture_height - 1) / 2;

//     // //using align corners=true, so apparently the incorrect way of doing it
//     // float ix=((cur_uv[0] + 1.f) / 2) * (texture_width - 1);
//     // float iy=((cur_uv[1] + 1.f) / 2) * (texture_height - 1);



//     //get the coordiantes of the neighbouring 4 pixels according to the wikipedia convention  https://en.wikipedia.org/wiki/Bilinear_interpolation
//     //get the coordiantes of the neighbouring 4 pixels according to the convention of https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c
//     int ix_nw = floor(ix);
//     int iy_nw = floor(iy);
//     int ix_ne = ix_nw + 1;
//     int iy_ne = iy_nw;
//     int ix_sw = ix_nw;
//     int iy_sw = iy_nw + 1;
//     int ix_se = ix_nw + 1;
//     int iy_se = iy_nw + 1;

//     //we calculate the weights here so as to also splat even though we are a bit out of border
//     float nw = (ix_se - ix)    * (iy_se - iy);
//     float ne = (ix    - ix_sw) * (iy_sw - iy);
//     float sw = (ix_ne - ix)    * (iy    - iy_ne);
//     float se = (ix - ix_nw) * (iy - iy_nw);


//     //coordinates and if the coordinate was clipped, also set the weight to zero
//     CLIP_COORDINATES_AND_WEIGHTS(ix_nw, nw, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_nw, nw, texture_height);
//     CLIP_COORDINATES_AND_WEIGHTS(ix_ne, ne, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_ne, ne, texture_height);
//     CLIP_COORDINATES_AND_WEIGHTS(ix_sw, sw, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_sw, sw, texture_height);
//     CLIP_COORDINATES_AND_WEIGHTS(ix_se, se, texture_width);
//     CLIP_COORDINATES_AND_WEIGHTS(iy_se, se, texture_height);


//     //GRAD TEXTURE
//     //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
//     float* d_nw_val = grad_texture + ix_nw*nr_channels_texture + iy_nw*texture_width*nr_channels_texture;
//     float* d_ne_val = grad_texture + ix_ne*nr_channels_texture + iy_ne*texture_width*nr_channels_texture;
//     float* d_sw_val = grad_texture + ix_sw*nr_channels_texture + iy_sw*texture_width*nr_channels_texture;
//     float* d_se_val = grad_texture + ix_se*nr_channels_texture + iy_se*texture_width*nr_channels_texture;

//     //get the weigthings of the pixels
//     // float denom = 1.0/( (x2 - x1)*(y2-y1)  + 1e-7 ); //the denominator is mostly just to normalize for the size of the pixel but we can assume a pixel of size 1 and just drop this whole term
//     // float wq11=(x2-x)*(y2-y);
//     // float wq21=(x-x1)*(y2-y);
//     // float wq12=(x2-x)*(y-y1);
//     // float wq22=(x-x1)*(y-y1);



//     //splat onto the 4 pixels the weighted cur_val
//     for(int i=0; i<nr_channels_texture; i++){
//         atomicAdd(d_nw_val + i, cur_grad_val[i]*nw );
//         atomicAdd(d_ne_val + i, cur_grad_val[i]*ne );
//         atomicAdd(d_sw_val + i, cur_grad_val[i]*sw );
//         atomicAdd(d_se_val + i, cur_grad_val[i]*se );
//     }




//     //GRAD UV
//     //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
//     const float* nw_val = texture + ix_nw*nr_channels_texture + iy_nw*texture_width*nr_channels_texture;
//     const float* ne_val = texture + ix_ne*nr_channels_texture + iy_ne*texture_width*nr_channels_texture;
//     const float* sw_val = texture + ix_sw*nr_channels_texture + iy_sw*texture_width*nr_channels_texture;
//     const float* se_val = texture + ix_se*nr_channels_texture + iy_se*texture_width*nr_channels_texture;
//     float grad_u=0;
//     float grad_v=0;
//     for(int i=0; i<nr_channels_texture; i++){
//         grad_u-=nw_val[i] * (iy_se - iy)*cur_grad_val[i];
//         grad_u+=ne_val[i] * (iy_sw - iy)*cur_grad_val[i];
//         grad_u-=sw_val[i] * (iy - iy_ne)*cur_grad_val[i];
//         grad_u+=se_val[i] * (iy - iy_nw)*cur_grad_val[i];

//         grad_v-=nw_val[i] * (ix_se - ix) *cur_grad_val[i];
//         grad_v-=ne_val[i] * (ix - ix_sw)*cur_grad_val[i];
//         grad_v+=sw_val[i] * (ix_ne - ix)*cur_grad_val[i];
//         grad_v+=se_val[i] * (ix - ix_nw)*cur_grad_val[i];
//     }


//     //unnormalize the grad_uv back to the [-1.1] constraint
//     // https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c
//     grad_u = grad_u * (texture_width - 1) *0.5;
//     grad_v = grad_v * (texture_height - 1) *0.5;

//     //put them in the tensor
//     cur_grad_uv[0]=grad_u;
//     cur_grad_uv[1]=grad_v;


// }


// Registers CUDA implementations for mymuladd, mymul, myadd_out
// TORCH_LIBRARY_IMPL(splat_slice, CUDA, m) {
//   m.impl("splat_to_texture_cuda", &splat_to_texture_cuda);
// }



// torch::Tensor splat_to_texture_cuda(
//     torch::Tensor& values_tensor, 
//     torch::Tensor& uv_tensor,
//     const int texture_height, 
//     const int texture_width
//     ) {


//     int nr_values=values_tensor.size(0);
//     int val_dim=values_tensor.size(1);
//     // int nr_channels_texture = val_dim+1; // we have a +1 because we store also a homogeneous value
//     int nr_channels_texture = val_dim; // assume we already have a homogeneous coordinate

//     torch::Tensor texture = torch::zeros({ texture_height, texture_width, nr_channels_texture }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

//     dim3 blocks((nr_values - 1) / BLOCK_SIZE + 1, 1, 1);
//     dim3 blockSize(BLOCK_SIZE, 1, 1);

//     // AT_DISPATCH_FLOATING_TYPES(values_tensor.type(), "splat_to_texture_kernel", ([&] {
//     // splat_to_texture_kernel<scalar_t><<<blocks, blockSize>>>(
//     //     nr_values,
//     //     texture.data<scalar_t>(),
//     //     values_tensor.data<scalar_t>(),
//     //     uv_tensor.data<scalar_t>(),
//     //    );
//     // }));

//     splat_to_texture_kernel<<<blocks, blockSize>>>(
//         nr_values,
//         val_dim, 
//         texture_height,
//         texture_width,
//         texture.data_ptr<float>(),
//         values_tensor.data_ptr<float>(),
//         uv_tensor.data_ptr<float>()
//     );


//     return texture;

// }


__device__ inline int start_index(int a, int b, int c) {
    return (int)floor((float)(a * c) / b);
}

__device__ inline int end_index(int a, int b, int c) {
    return (int)ceil((float)((a + 1) * c) / b);
}

template<typename scalar_t>
__global__ void FastInpaintingKernel_Down(dim3 virtual_size,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> mask,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> data,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskLow,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataLow)
{
    const int H = mask.size(1);
    const int W = mask.size(2);
    const int oH = H / 2;
    const int oW = W / 2;
    const int C = data.size(1);
    CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of low resolution
    {
        int N = 0;
        scalar_t d[MAX_CHANNELS] = { 0 };
        for (int jj = start_index(j, oW, W); jj < end_index(j, oW, W); ++jj)
            for (int ii = start_index(i, oH, H); ii < end_index(i, oH, H); ++ii)
            {
                if (mask[b][ii][jj] >= 0.5)
                {
                    N++;
                    for (int c = 0; c < C; ++c)
                        d[c] += data[b][c][ii][jj];
                }
            }
        maskLow[b][i][j] = N > 0 ? 1 : 0;
        for (int c = 0; c < C; ++c)
            dataLow[b][c][i][j] = N > 0 ? d[c] / N : 0;
    }
    CUMAT_KERNEL_3D_LOOP_END
}

template<typename scalar_t>
__global__ void FastInpaintingKernel_Up(dim3 virtual_size,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> mask,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> data,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskLow,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataLow,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskHigh,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataHigh)
{
    const int H = mask.size(1);
    const int W = mask.size(2);
    const int oH = H / 2;
    const int oW = W / 2;
    const int C = data.size(1);
    CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of low resolution
    {
        if (mask[b][i][j] >= 0.5)
        {
            //copy unchanged
            maskHigh[b][i][j] = 1;
            for (int c = 0; c < C; ++c)
                dataHigh[b][c][i][j] = data[b][c][i][j];
        }
        else
        {
            //interpolate from low resolution (bilinear)
            //get neighbor offsets
            int io = i % 2 == 0 ? -1 : +1;
            int jo = j % 2 == 0 ? -1 : +1;
            //accumulate
            scalar_t N = 0;
            scalar_t d[MAX_CHANNELS] = { 0 };
#define ITEM(ii,jj,w)													\
if ((ii)>=0 && (jj)>=0 && (ii)<oH && (jj)<oW && maskLow[b][(ii)][(jj)]>=0.5) {	\
    N += w;															\
    for (int c = 0; c < C; ++c) d[c] += w * dataLow[b][c][(ii)][(jj)];		\
}
            ITEM(i / 2, j / 2, 0.75f*0.75f);
            ITEM(i / 2 + io, j / 2, 0.25f*0.75f);
            ITEM(i / 2, j / 2 + jo, 0.25f*0.75f);
            ITEM(i / 2 + io, j / 2 + jo, 0.25f*0.25f);
#undef ITEM
            //write output
            maskHigh[b][i][j] = N > 0 ? 1 : 0;
            for (int c = 0; c < C; ++c)
                dataHigh[b][c][i][j] = N > 0 ? d[c] / N : 0;
        }
    }
    CUMAT_KERNEL_3D_LOOP_END
}




std::tuple<torch::Tensor, torch::Tensor>
		push_pull_inpaint_recursion_cuda(
			const torch::Tensor& mask,
			const torch::Tensor& data)
	{
		int64_t B = data.size(0);
		int64_t C = data.size(1);
		int64_t H = data.size(2);
		int64_t W = data.size(3);

		if (H <= 1 && W <= 1)
			return std::make_tuple(mask, data); //end of recursion

		int64_t oH = H / 2;
		int64_t oW = W / 2;

		// prepare launching
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		//downsample
		torch::Tensor maskLow = torch::empty({ B, oH, oW }, mask.options());
		torch::Tensor dataLow = torch::empty({ B, C, oH, oW }, data.options());
		AT_DISPATCH_FLOATING_TYPES(data.type(), "FastInpaintingKernel_Down", ([&]
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
				oW, oH, B, FastInpaintingKernel_Down<scalar_t>);
			FastInpaintingKernel_Down<scalar_t>
				<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size,
					mask.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskLow.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataLow.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
		}));
		CUMAT_CHECK_ERROR();

		//recursion
		const auto tuple = push_pull_inpaint_recursion_cuda(maskLow, dataLow);
		const auto& maskLow2 = std::get<0>(tuple);
		const auto& dataLow2 = std::get<1>(tuple);

		//upsample
		torch::Tensor maskHigh = torch::empty({ B, H, W }, mask.options());
		torch::Tensor dataHigh = torch::empty({ B, C, H, W }, data.options());
		AT_DISPATCH_FLOATING_TYPES(data.type(), "FastInpaintingKernel_Up", ([&]
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
				W, H, B, FastInpaintingKernel_Up<scalar_t>);
			FastInpaintingKernel_Up<scalar_t>
				<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size,
					mask.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskLow2.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataLow2.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskHigh.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataHigh.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
		}));
		CUMAT_CHECK_ERROR();

		//done
		return std::make_tuple(maskHigh, dataHigh);
	}