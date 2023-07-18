cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(drums)

list(APPEND CMAKE_PREFIX_PATH "E:/GitHub/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(drums drums.cpp)
target_link_libraries(drums "${TORCH_LIBRARIES}")
set_property(TARGET drums PROPERTY CXX_STANDARD 14)

target_precompile_headers(${PROJECT_NAME} 
	PRIVATE 
	<torch/torch.h>
	<string>
	<thread>
	<vector>)
