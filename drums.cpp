#include <iostream>
#include <torch/torch.h>

struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

int main() {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
    std::cout << "Using CUDA." << std::endl;
    device = torch::kCUDA;
    }

    Net net(2, 3);
    net.to(device);

    std::cout << net << std::endl;

    std::cout << net.forward(torch::ones({4, 2}).to(device)) << std::endl;
}