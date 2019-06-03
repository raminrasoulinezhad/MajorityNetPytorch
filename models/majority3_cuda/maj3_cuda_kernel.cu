//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Ramins sources:
//		1- to shrink memory footprints:
//			-- https://pytorch.org/cppdocs/api/program_listing_file_torch_csrc_api_include_torch_types.h.html?highlight=kfloat
//			-- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
//			-- https://jhui.github.io/2017/03/06/CUDA/
//			-- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
//			-- https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
//		2- lunching the cuda kernels:
//			-- // source : http://www.icl.utk.edu/~mgates3/docs/cuda.html
//		3- sample about "PackedTensorAccessor"
//			-- https://github.com/pytorch/pytorch/issues/13018
//			-- https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/test/cuda_packedtensoraccessor_test.cu
//			-- https://fossies.org/linux/pytorch/aten/src/ATen/native/cuda/FractionalMaxPool2d.cu
// 		4- backward computations:
//			-- https://github.com/webgpu/ece508-convlayer/blob/master/src/main.cu
//		5- Pytorch CPP extention:
//			-- Doc:	https://pytorch.org/cppdocs/index.html
//		6- BNN.pytorch
//			-- https://github.com/itayhubara/BinaryNet.pytorch
//		7- Cuda Toturial
//			-- https://pytorch.org/tutorials/advanced/cpp_extension.html
//			-- https://pytorch.org/tutorials/advanced/cpp_extension.html
//		8- accessing a data in tensor:
//			-- https://discuss.pytorch.org/t/c-aten-pick-integer-value-from-a-tensor/27840
//			-- https://pytorch.org/cppdocs/notes/tensor_basics.html
//			-- https://discuss.pytorch.org/t/basic-tensor-manipulation-in-c/27383
//			-- 
//////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

//////////////////////////////////////////////////////////////////////////////
// bare CUDA functions
//////////////////////////////////////////////////////////////////////////////

namespace {

template <typename scalar_t>
__global__ void maj3_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input_paded,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weights_padded,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
    torch::PackedTensorAccessor<int8_t,6,torch::RestrictPtrTraits,size_t> inter
	){

	const auto David = 2.25;

	const int c = blockIdx.x;
	const int b = blockIdx.y;
	const int w = threadIdx.x;
	const int h = threadIdx.y;

	const int c_in_size = input_paded.size(3);
	const int k_w_size = weights_padded.size(1);
	const int k_h_size = weights_padded.size(2);

  	auto pop = 0.0;

  	int i, j, k;
 	for (i = 0; i < c_in_size; i = i + 1){
		for (j = 0; j < k_h_size; j = j + 1){
  			int temp = 0;
  			for (k = 0; k < k_w_size; k = k + 1){
  				temp = temp + input_paded[b][w+k][h+j][i] * weights_padded[c][k][j][i];
  			}
  			if (temp > 0){
  				pop = pop +1;
  			}

  			if (abs(temp) < 2.0){
  				inter[b][w][h][j][i][c] = 1;
  			}
  		}
  	}

  	output[b][w][h][c] = (2 * pop - (k_h_size * c_in_size)) * David;
}



template <typename scalar_t>
__global__ void maj3_cuda_backward_kernel_d_input(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_output_padded,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<int8_t,6,torch::RestrictPtrTraits,size_t> inter_padded,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_input) {	

    const auto David = 2.25;

  	const int b 	= blockIdx.y;
  	const int w_in 	= threadIdx.x;
  	const int h_in 	= threadIdx.y;
  	const int c_in 	= blockIdx.x;

  	const int c_out_size = weights.size(0);
  	const int k_w_size = weights.size(1);
  	const int k_h_size = weights.size(2);  	

  	auto temp = 0.0;
  	int i, j, k;
 	for (i = 0; i < k_w_size; i = i + 1){
		for (j = 0; j < k_h_size; j = j + 1){
  			for (k = 0; k < c_out_size; k = k + 1){
  				//////// the actual value is required from d_outputs
  				// temp_d_output = d_output_padded[b][w_in+i][h_in+j][k][i][j][c_in];
  				// Using inter tensor by broadcasting its values
  				// inter indexing: inter[b][w_out][h_out][kh][c_in][c_out]

  				auto temp_d_output = d_output_padded[b][w_in+i][h_in+j][k] * David * inter_padded[b][w_in+i][h_in+j][j][c_in][k];
  				auto temp_weight = weights[k][k_w_size-i-1][k_h_size-j-1][c_in];
  				temp = temp + temp_d_output * temp_weight;  								
  			}
  			
  		}
  	}
  	d_input[b][w_in][h_in][c_in] = temp;
}


template <typename scalar_t>
__global__ void maj3_cuda_backward_kernel_d_weights(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_output,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input_padded,
    const torch::PackedTensorAccessor<int8_t,6,torch::RestrictPtrTraits,size_t> inter,    
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_weights) {	

    const auto David = 2.25;

  	const int c_in	= blockIdx.x;
  	const int k_w	= blockIdx.y;
  	const int k_h	= blockIdx.z;
  	const int c_out  = threadIdx.x;  	

  	const int b_size = d_output.size(0);
  	const int w_out_size = d_output.size(1);
  	const int h_out_size = d_output.size(2);  	

  	auto temp = 0.0;
  	int i, j, k;
 	for (i = 0; i < w_out_size; i = i + 1){
		for (j = 0; j < h_out_size; j = j + 1){
  			for (k = 0; k < b_size; k = k + 1){
  				// inter indexing: inter[b][w_out][h_out][kh][c_in][c_out]			
  				auto temp_d_output = d_output[k][i][j][c_out] * David * inter[k][i][j][k_h][c_in][c_out];
  				auto temp_input = input_padded[k][i+k_w][j+k_h][c_in];
  				temp = temp + temp_d_output * temp_input;
  			}
  			
  		}
  	}
  	d_weights[c_out][k_w][k_h][c_in] = temp;
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


template <typename scalar_t>
__global__ void maj3_cuda_forward_kernel_NBP_v1(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input_paded,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weights_padded,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output){

	const auto David = 2.25;

	const int c = blockIdx.x;
	const int b = blockIdx.y;
	const int w = threadIdx.x;
	const int h = threadIdx.y;

	const int c_in_size = input_paded.size(3);
	const int k_w_size = weights_padded.size(1);
	const int k_h_size = weights_padded.size(2);

  	auto pop = 0.0;
  	int i, j, k;
 	for (i = 0; i < c_in_size; i = i + 1){
		for (j = 0; j < k_h_size; j = j + 1){
  			int temp = 0;
  			for (k = 0; k < k_w_size; k = k + 1){
  				temp = temp + input_paded[b][w+k][h+j][i] * weights_padded[c][k][j][i];
  			}
  			if (temp > 0){
  				pop = pop +1;
  			}
  		}
  	}
  	output[b][w][h][c] = (2 * pop - (k_h_size * c_in_size)) * David;
}


template <typename scalar_t>
__global__ void maj3_cuda_forward_kernel_NBP_v2(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input_paded,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weights_padded,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output){

	const auto David = 2.25;

	const int c = blockIdx.x;
	const int b = blockIdx.y;
	const int w = threadIdx.x;
	const int h = threadIdx.y;

	const int c_in_size = input_paded.size(3);
	const int k_w_size = weights_padded.size(1);
	const int k_h_size = weights_padded.size(2);

	//const int buffer_size = c_in_size * k_h_size * k_w_size;
	//const int buffer_size = 128 * 3 * 3;
	//scalar_t *input_Reg = new scalar_t [buffer_size];
	//scalar_t *weight_Reg = new scalar_t [buffer_size];

	scalar_t input_Reg [128*3*3];
	__shared__ scalar_t weight_Reg [128*3*3];	
	
	int i, j, k;
	for (k = 0; k < k_w_size; k = k + 1){
		for (j = 0; j < k_h_size; j = j + 1){
  			for (i = 0; i < c_in_size; i = i + 1){
  				int index = (i * k_h_size * k_w_size) + (j * k_w_size) + k;
  				//input_Reg[index]  = input_paded[b][w+k][h+j][i];
  				if ((w==0)&(h==0))
  					weight_Reg[index] = weights_padded[c][k][j][i];
  			}
  		}
  	}
  	__syncthreads();

  	auto pop = 0.0;
 	for (i = 0; i < (c_in_size); i = i + 1){
 		for (j = 0; j < k_h_size; j = j + 1){
			int temp = 0;
			for (k = 0; k < k_w_size; k = k + 1){

				int index = (i * k_h_size * k_w_size) + (j * k_w_size) + k;
				temp = temp + input_paded[b][w+k][h+j][i] * weight_Reg[index];

				//temp = temp + input_Reg[i * k_w_size + k] * weight_Reg[i * k_w_size + k];

				///auto temp1 = input_Reg[0];
				///auto temp2 = weight_Reg[0];
				///temp = temp + temp1 * temp2;
			}
			if (temp > 0){
				pop = pop +1;
			}
		}
  	}

  	//delete input_Reg;
  	//delete weight_Reg;
  	
  	output[b][w][h][c] = (2 * pop - (k_h_size * c_in_size)) * David;
}


template <typename scalar_t>
__global__ void maj3_cuda_forward_kernel_NBP_v3(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input_paded,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weights_padded,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output){

	const auto David = 2.25;

	// TO CALL
	//const dim3 blocks(w_out, h_out, b);
	//const dim3 threads(c_out, 1, 1);
	const int c = threadIdx.x;
	const int w = blockIdx.x;
	const int h = blockIdx.y;
	const int b = blockIdx.z;

	const int c_in_size = input_paded.size(3);
	const int k_w_size = weights_padded.size(1);
	const int k_h_size = weights_padded.size(2);

  	auto pop = 0.0;
  	int i, j, k;
 	for (i = 0; i < c_in_size; i = i + 1){
		for (j = 0; j < k_h_size; j = j + 1){
  			int temp = 0;
  			for (k = 0; k < k_w_size; k = k + 1){
  				temp = temp + input_paded[b][w+k][h+j][i] * weights_padded[c][k][j][i];
  			}
  			if (temp > 0){
  				pop = pop +1;
  			}
  		}
  	}
  	output[b][w][h][c] = (2 * pop - (k_h_size * c_in_size)) * David;
}


template <typename scalar_t>
__global__ void maj3_cuda_forward_kernel_NBP_v4(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input_paded,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weights_padded,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output){

	const auto David = 2.25;

	// TO CALL
	//const int block_divider = 1024/c_out;
	//const dim3 blocks(w_out, h_out, b/block_divider);
	//const dim3 threads(c_out, block_divider, 1);

	const int c = threadIdx.x;
	const int b_t = threadIdx.y;
	const int w = blockIdx.x;
	const int h = blockIdx.y;
	const int b_b = blockIdx.z;
	const int b = b_b * blockDim.y + b_t;

	const int c_in_size = input_paded.size(3);
	const int k_w_size = weights_padded.size(1);
	const int k_h_size = weights_padded.size(2);

  	auto pop = 0.0;
  	int i, j, k;
 	for (i = 0; i < c_in_size; i = i + 1){
		for (j = 0; j < k_h_size; j = j + 1){
  			int temp = 0;
  			for (k = 0; k < k_w_size; k = k + 1){
  				temp = temp + input_paded[b][w+k][h+j][i] * weights_padded[c][k][j][i];
  			}
  			if (temp > 0){
  				pop = pop +1;
  			}
  		}
  	}
  	output[b][w][h][c] = (2 * pop - (k_h_size * c_in_size)) * David;
}



template <typename scalar_t>
__global__ void maj3_cuda_backward_kernel_d_input_NBP(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_output_padded,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weights,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_input) {	

    const auto David = 2.25;

  	const int b 	= blockIdx.y;
  	const int w_in 	= threadIdx.x;
  	const int h_in 	= threadIdx.y;
  	const int c_in 	= blockIdx.x;

  	const int c_out_size = weights.size(0);
  	const int k_w_size = weights.size(1);
  	const int k_h_size = weights.size(2);
  	
  	auto temp = 0.0;
  	int i, j, k;
 	for (i = 0; i < k_w_size; i = i + 1){
		for (j = 0; j < k_h_size; j = j + 1){
  			for (k = 0; k < c_out_size; k = k + 1){
  				auto temp_d_output = d_output_padded[b][w_in+i][h_in+j][k] * David;
  				auto temp_weight = weights[k][k_w_size-i-1][k_h_size-j-1][c_in];
  				temp = temp + temp_d_output * temp_weight;  								
  			}
  			
  		}
  	}
  	d_input[b][w_in][h_in][c_in] = temp;
}



template <typename scalar_t>
__global__ void maj3_cuda_backward_kernel_d_weights_NBP(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_output,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input_padded,   
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_weights) {	

    const auto David = 2.25;

  	const int c_in	= blockIdx.x;
  	const int k_w	= blockIdx.y;
  	const int k_h	= blockIdx.z;
  	const int c_out	= threadIdx.x;  

  	const int b_size = d_output.size(0);
  	const int w_out_size = d_output.size(1);
  	const int h_out_size = d_output.size(2);
  	
  	auto temp = 0.0;
  	int i, j, k;
 	for (i = 0; i < w_out_size; i = i + 1){
		for (j = 0; j < h_out_size; j = j + 1){
  			for (k = 0; k < b_size; k = k + 1){
  				auto temp_d_output = d_output[k][i][j][c_out] * David;
  				auto temp_input = input_padded[k][i+k_w][j+k_h][c_in];
  				temp = temp + temp_d_output * temp_input;
  			}
  		}
  	}
  	d_weights[c_out][k_w][k_h][c_in] = temp;
}

template <typename scalar_t>
__global__ void maj3_cuda_forward_kernel_NBP_noPad(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input_paded,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weights_padded,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output){

	const auto David = 2.25;

	const int c = blockIdx.x;
	const int b = blockIdx.y;
	const int w = threadIdx.x;
	const int h = threadIdx.y;

	const int c_in_size = input_paded.size(3);
	const int k_w_size = weights_padded.size(1);
	const int k_h_size = weights_padded.size(2);

  	auto pop = 0.0;
  	int i, j, k;
 	for (i = 0; i < c_in_size; i = i + 1){
		for (j = 0; j < k_h_size; j = j + 1){
  			int temp = 0;
  			for (k = 0; k < k_w_size; k = k + 1){
  				temp = temp + input_paded[b][w+k][h+j][i] * weights_padded[c][k][j][i];
  			}
  			if (temp > 0){
  				pop = pop +1;
  			}
  		}
  	}
  	output[b][w][h][c] = (2 * pop - (k_h_size * c_in_size)) * David;
}



} // namespace




//////////////////////////////////////////////////////////////////////////////
// CUDA managers
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
// Pure majority forward and backpropagation 


std::vector<torch::Tensor> maj3_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights) {

	const int b = input.size(0);
	const int w_in = input.size(1);
	const int h_in = input.size(2);
	const int c_in = input.size(3);
	
	const int c_out = weights.size(0);
	const int k_w = weights.size(1);
	const int k_h = weights.size(2);

	const int w_out = w_in;
	const int h_out = h_in;
  
	auto input_paded = at::constant_pad_nd(input, {0,0,1,1,1,1,0,0}, -1.0);   
	auto output = torch::zeros({b, w_out, h_out, c_out}, torch::CUDA(at::kFloat));
  	auto inter = torch::zeros({b, w_out, h_out, k_h, c_in, c_out}, torch::CUDA(at::kChar));


	const dim3 blocks(c_out, b, 1);
	const dim3 threads(w_out, h_out, 1);

	AT_DISPATCH_FLOATING_TYPES(input_paded.type(), "maj3_forward_cuda", ([&] {maj3_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
	    input_paded.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    inter.packed_accessor<int8_t,6,torch::RestrictPtrTraits,size_t>());
	}));

	// L1 & L2 & L3
	// auto input_paded = at::constant_pad_nd(input, {0,0,1,1,1,1,0,0}, -1.0);                   
	// auto input_unfolded = at::_th_unfold(at::_th_unfold(input_paded, 1, k_w, 1), 2, k_h, 1).reshape({-1, k_w*k_h*c_in});
	// auto inter_maj3 = at::sum(at::einsum("bj,aj->baj", {input_unfolded, weights.reshape({-1, k_w*k_h*c_in})}).reshape({-1, k_w, k_h*c_in}), 1);
	// L123
	// auto inter_maj3 = at::sum(at::einsum("bj,aj->baj", {at::_th_unfold(at::_th_unfold(at::constant_pad_nd(input, {0,0,1,1,1,1,0,0}, -1.0), 1, k_w, 1), 2, k_h, 1).reshape({-1, k_w*k_h*c_in}), weights.reshape({-1, k_w*k_h*c_in})}).reshape({-1, k_w, k_h*c_in}), 1);

	// L4 & L5 (I replaced sum and mul)
	// auto pop = at::sum(at::_th_clamp(inter_maj3, -1, 1), 1);
	// auto output = at::_cast_Float(at::mul(pop, 2.25).reshape({-1, w_in, h_in, c_out}), 0);
	// L45
	// auto output = at::_cast_Float(at::mul(at::sum(at::_th_clamp(inter_maj3, -1, 1), 1), 2.25).reshape({-1, w_in, h_in, c_out}), 0);

	// L12345
	// auto output = at::_cast_Float(at::mul(at::sum(at::_th_clamp(at::sum(at::einsum("bj,aj->baj", {at::_th_unfold(at::_th_unfold(at::constant_pad_nd(input, {0,0,1,1,1,1,0,0}, -1.0), 1, k_w, 1), 2, k_h, 1).reshape({-1, k_w*k_h*c_in}), weights.reshape({-1, k_w*k_h*c_in})}).reshape({-1, k_w, k_h*c_in}), 1), -1, 1), 1), 2.25).reshape({-1, w_in, h_in, c_out}), 0);

	return {output, inter};
}


std::vector<torch::Tensor> maj3_cuda_backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor inter) {

	const int b = d_output.size(0);
	const int w_out = d_output.size(1);
	const int h_out = d_output.size(2);
	const int c_out = d_output.size(3);

	const int k_w = weights.size(1);
	const int k_h = weights.size(2);
	const int c_in = weights.size(3);

	const int w_in = input.size(1);
	const int h_in = input.size(2);

	auto d_input   = torch::zeros({b, w_in, h_in, c_in}, torch::CUDA(at::kFloat));
	auto d_weights = torch::zeros({c_out, k_w, k_h, c_in}, torch::CUDA(at::kFloat));

	auto d_output_padded = at::constant_pad_nd(d_output, {0,0,1,1,1,1,0,0}, 0.0);
	auto input_padded = at::constant_pad_nd(input, {0,0,1,1,1,1,0,0}, -1.0);
	auto inter_padded = at::constant_pad_nd(inter, {0,0,0,0,0,0,1,1,1,1,0,0}, 0.0); 

	const dim3 blocks_d_input(c_in, b, 1);
	const dim3 threads_d_input(w_out, h_out, 1);
	AT_DISPATCH_FLOATING_TYPES(d_output_padded.type(), "maj3_backward_cuda", ([&] {maj3_cuda_backward_kernel_d_input<scalar_t><<<blocks_d_input, threads_d_input>>>(
	    d_output_padded.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    inter_padded.packed_accessor<int8_t,6,torch::RestrictPtrTraits,size_t>(),
	    d_input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
	}));


	const dim3 blocks_d_weights(c_in, k_w, k_h);
	const dim3 threads_d_weights(c_out, 1, 1);
	AT_DISPATCH_FLOATING_TYPES(d_output_padded.type(), "maj3_backward_cuda", ([&] {maj3_cuda_backward_kernel_d_weights<scalar_t><<<blocks_d_weights, threads_d_weights>>>(
	    d_output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    input_padded.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    inter.packed_accessor<int8_t,6,torch::RestrictPtrTraits,size_t>(),
	    d_weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
	}));


	return {d_input, d_weights};
}


//////////////////////////////////////////////////////////////////////////////
// similar to normal Conv backpropagation and majority forward

std::vector<torch::Tensor> maj3_cuda_forward_NBP(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding) {

	const int b = input.size(0);
	const int w_in = input.size(1);
	const int h_in = input.size(2);
	const int c_in = input.size(3);
	
	const int c_out = weights.size(0);
	const int k_w = weights.size(1);
	const int k_h = weights.size(2);

	const int w_out = w_in;
	const int h_out = h_in;
  
  	//auto padding_a = padding.accessor<int32_t, 1>();
  	//auto pad = padding_a[0]; 

  	auto input_paded = at::constant_pad_nd(input, {0,0,1,1,1,1,0,0}, -1.0);   
	auto output = torch::zeros({b, w_out, h_out, c_out}, torch::CUDA(at::kFloat));

  	//if (pad == 0){
	//	return {output};
  	//	//auto output = torch::zeros({b, w_out-pad, h_out-pad, c_out}, torch::CUDA(at::kFloat));
	//}
	//else {
		//auto input_paded = at::constant_pad_nd(input, {0,0,pad,pad,pad,pad,0,0}, -1.0);   
		//auto output = torch::zeros({b, w_out, h_out, c_out}, torch::CUDA(at::kFloat));
	//}

	const dim3 blocks(c_out, b, 1);
	const dim3 threads(w_out, h_out, 1);
	AT_DISPATCH_FLOATING_TYPES(input_paded.type(), "maj3_forward_cuda", ([&] {maj3_cuda_forward_kernel_NBP_v1<scalar_t><<<blocks, threads>>>(
	    input_paded.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
	}));

	//const dim3 blocks(w_out, h_out, b);
	//const dim3 threads(c_out, 1, 1);
	//AT_DISPATCH_FLOATING_TYPES(input_paded.type(), "maj3_forward_cuda", ([&] {maj3_cuda_forward_kernel_NBP_v3<scalar_t><<<blocks, threads>>>(
	//    input_paded.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	//    weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	//    output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
	//}));

	return {output};
}

std::vector<torch::Tensor> maj3_cuda_backward_NBP(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding) {

	const int b = d_output.size(0);
	const int w_out = d_output.size(1);
	const int h_out = d_output.size(2);
	const int c_out = d_output.size(3);

	const int k_w = weights.size(1);
	const int k_h = weights.size(2);
	const int c_in = weights.size(3);

	const int w_in = input.size(1);
	const int h_in = input.size(2);

	auto d_input   = torch::zeros({b, w_in, h_in, c_in}, torch::CUDA(at::kFloat));
	auto d_weights = torch::zeros({c_out, k_w, k_h, c_in}, torch::CUDA(at::kFloat));

	auto d_output_padded = at::constant_pad_nd(d_output, {0,0,1,1,1,1,0,0}, 0.0);
	auto input_padded = at::constant_pad_nd(input, {0,0,1,1,1,1,0,0}, -1.0);

	const dim3 blocks_d_input(c_in, b, 1);
	const dim3 threads_d_input(w_out, h_out, 1);
	AT_DISPATCH_FLOATING_TYPES(d_output_padded.type(), "maj3_backward_cuda", ([&] {maj3_cuda_backward_kernel_d_input_NBP<scalar_t><<<blocks_d_input, threads_d_input>>>(
	    d_output_padded.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    d_input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
	}));

	const dim3 blocks_d_weights(c_in, k_w, k_h);
	const dim3 threads_d_weights(c_out, 1, 1);
	AT_DISPATCH_FLOATING_TYPES(d_output_padded.type(), "maj3_backward_cuda", ([&] {maj3_cuda_backward_kernel_d_weights_NBP<scalar_t><<<blocks_d_weights, threads_d_weights>>>(
	    d_output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    input_padded.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    d_weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
	}));

	return {d_input, d_weights};
}


//////////////////////////////////////////////////////////////////////////////
// similar to normal Conv backpropagation and majority forward

std::vector<torch::Tensor> maj3_cuda_forward_NBP_noPad(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding) {

	const int b = input.size(0);
	const int w_in = input.size(1);
	const int h_in = input.size(2);
	const int c_in = input.size(3);
	
	const int c_out = weights.size(0);
	const int k_w = weights.size(1);
	const int k_h = weights.size(2);

	const int w_out = w_in-2;
	const int h_out = h_in-2;
 
	auto output = torch::zeros({b, w_out, h_out, c_out}, torch::CUDA(at::kFloat));

	const dim3 blocks(c_out, b, 1);
	const dim3 threads(w_out, h_out, 1);
	AT_DISPATCH_FLOATING_TYPES(input.type(), "maj3_forward_cuda", ([&] {maj3_cuda_forward_kernel_NBP_noPad<scalar_t><<<blocks, threads>>>(
	    input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
	}));

	return {output};
}

std::vector<torch::Tensor> maj3_cuda_backward_NBP_noPad(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding) {

	const int b = d_output.size(0);
	const int w_out = d_output.size(1);
	const int h_out = d_output.size(2);
	const int c_out = d_output.size(3);

	const int k_w = weights.size(1);
	const int k_h = weights.size(2);
	const int c_in = weights.size(3);

	const int w_in = input.size(1);
	const int h_in = input.size(2);

	auto d_input   = torch::zeros({b, w_in, h_in, c_in}, torch::CUDA(at::kFloat));
	auto d_weights = torch::zeros({c_out, k_w, k_h, c_in}, torch::CUDA(at::kFloat));

	auto d_output_padded = at::constant_pad_nd(d_output, {0,0,2,2,2,2,0,0}, 0.0);
	auto input_padded = at::constant_pad_nd(input, {0,0,1,1,1,1,0,0}, -1.0);

	const dim3 blocks_d_input(c_in, b, 1);
	const dim3 threads_d_input(w_out, h_out, 1);
	AT_DISPATCH_FLOATING_TYPES(d_output_padded.type(), "maj3_backward_cuda", ([&] {maj3_cuda_backward_kernel_d_input_NBP<scalar_t><<<blocks_d_input, threads_d_input>>>(
	    d_output_padded.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    d_input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
	}));

	const dim3 blocks_d_weights(c_in, k_w, k_h);
	const dim3 threads_d_weights(c_out, 1, 1);
	AT_DISPATCH_FLOATING_TYPES(d_output_padded.type(), "maj3_backward_cuda", ([&] {maj3_cuda_backward_kernel_d_weights_NBP<scalar_t><<<blocks_d_weights, threads_d_weights>>>(
	    d_output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
	    d_weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
	}));

	return {d_input, d_weights};
}