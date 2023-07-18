#include <iostream>
#include <torch/torch.h>

// next: pre-compiled headers

int main() {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
    std::cout << "Using CUDA." << std::endl;
    device = torch::kCUDA;
    }

    torch::Tensor tensor = torch::eye(3).to(device);
    std::cout << tensor << std::endl;
}