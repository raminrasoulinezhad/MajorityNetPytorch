#include <torch/extension.h>
#include <vector>
#define UnderDevelopment 0
//////////////////////////////////////////////////////////////////////////////
// CUDA forward declarations
std::vector<torch::Tensor> maj3_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights);

std::vector<torch::Tensor> maj3_cuda_backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor inter);
//-------------------------------------------------
std::vector<torch::Tensor> maj3_cuda_forward_NBP(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding);

std::vector<torch::Tensor> maj3_cuda_backward_NBP(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding);
//-------------------------------------------------
std::vector<torch::Tensor> maj3_cuda_forward_NBP_noPad(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding);

std::vector<torch::Tensor> maj3_cuda_backward_NBP_noPad(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding);

//////////////////////////////////////////////////////////////////////////////
// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//////////////////////////////////////////////////////////////////////////////
std::vector<torch::Tensor> maj3_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding){

	if (UnderDevelopment){	
		CHECK_INPUT(input);
		CHECK_INPUT(weights);
		CHECK_INPUT(padding);
	}
  	return maj3_cuda_forward(input, weights);
}
std::vector<torch::Tensor> maj3_backward(
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
	return maj3_cuda_backward(
		d_output,
		input,
		weights,
		inter);
}
//-------------------------------------------------
std::vector<torch::Tensor> maj3_forward_NBP(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding){

	if (UnderDevelopment){	
		CHECK_INPUT(input);
		CHECK_INPUT(weights);
		CHECK_INPUT(padding);
	}
	return maj3_cuda_forward_NBP(input, weights, padding);
}
std::vector<torch::Tensor> maj3_backward_NBP(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding){

	if (UnderDevelopment){	
		CHECK_INPUT(d_output);
		CHECK_INPUT(input);
		CHECK_INPUT(weights);
		CHECK_INPUT(padding);
	}
  	return maj3_cuda_backward_NBP(d_output, input, weights, padding);
}
//-------------------------------------------------
std::vector<torch::Tensor> maj3_forward_NBP_noPad(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding){

	if (UnderDevelopment){	
		CHECK_INPUT(input);
		CHECK_INPUT(weights);
		CHECK_INPUT(padding);
	}
	return maj3_cuda_forward_NBP_noPad(input, weights, padding);
}
std::vector<torch::Tensor> maj3_backward_NBP_noPad(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor padding){

	if (UnderDevelopment){	
		CHECK_INPUT(d_output);
		CHECK_INPUT(input);
		CHECK_INPUT(weights);
		CHECK_INPUT(padding);
	}
  	return maj3_cuda_backward_NBP_noPad(d_output, input, weights, padding);
}

//////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &maj3_forward, "maj3 forward (CUDA)");
  m.def("backward", &maj3_backward, "maj3 backward (CUDA)");
  m.def("forward_NBP", &maj3_forward_NBP, "maj3 forward NBP (CUDA)");
  m.def("backward_NBP", &maj3_backward_NBP, "maj3 backward NBP (CUDA)");
  m.def("forward_NBP_noPad", &maj3_forward_NBP_noPad, "maj3 forward NBP (CUDA)");
  m.def("backward_NBP_noPad", &maj3_backward_NBP_noPad, "maj3 backward NBP (CUDA)");
}


