#include <torch/extension.h>
#include <vector>
#define UnderDevelopment 0

//////////////////////////////////////////////////////////////////////////////
//  NBP: Normal Back Propagation, means using normal Convolution layer backpropagation rather than using exact backpropagation function 

// CUDA forward declarations
std::vector<torch::Tensor> maj_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights);

std::vector<torch::Tensor> maj_cuda_backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor inter);
//-------------------------------------------------
std::vector<torch::Tensor> maj_cuda_forward_NBP(
    torch::Tensor input,
    torch::Tensor weights,
    int  padding);

std::vector<torch::Tensor> maj_cuda_backward_NBP(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    int padding);

//////////////////////////////////////////////////////////////////////////////
// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//////////////////////////////////////////////////////////////////////////////
std::vector<torch::Tensor> maj_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding){

	if (UnderDevelopment){	
		CHECK_INPUT(input);
		CHECK_INPUT(weights);
		CHECK_INPUT(padding);
	}
  	return maj_cuda_forward(input, weights);
}
std::vector<torch::Tensor> maj_backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor inter,
    torch::Tensor padding){

	if (UnderDevelopment){	
		CHECK_INPUT(d_output);
		CHECK_INPUT(input);
		CHECK_INPUT(weights);
		CHECK_INPUT(inter);
		CHECK_INPUT(padding);
	}
	return maj_cuda_backward(
		d_output,
		input,
		weights,
		inter);
}
//-------------------------------------------------
std::vector<torch::Tensor> maj_forward_NBP(
    torch::Tensor input,
    torch::Tensor weights,
    int  padding){

	if (UnderDevelopment){	
		CHECK_INPUT(input);
		CHECK_INPUT(weights);
	}

    return maj_cuda_forward_NBP(input, weights, padding);
}
std::vector<torch::Tensor> maj_backward_NBP(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    int padding){

	if (UnderDevelopment){	
		CHECK_INPUT(d_output);
		CHECK_INPUT(input);
		CHECK_INPUT(weights);
	}
  	return maj_cuda_backward_NBP(d_output, input, weights, padding);
}

//////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &maj_forward, "maj forward (CUDA)");
  m.def("backward", &maj_backward, "maj backward (CUDA)");
  m.def("forward_NBP", &maj_forward_NBP, "maj forward NBP (CUDA)");
  m.def("backward_NBP", &maj_backward_NBP, "maj backward NBP (CUDA)");
}


