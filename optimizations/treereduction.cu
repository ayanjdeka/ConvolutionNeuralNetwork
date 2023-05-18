#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

//THIS IS THE TREE REDUCTION CODE
#define TILE_WIDTH 16
__constant__ float Mask[10000];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    int shared_width = TILE_WIDTH + K - 1;
    extern __shared__ float sharedMemory[];
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) Mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define sharedMemTree(i2, i1, i0) sharedMemory[i2 * TILE_WIDTH * Channel + i1 * Channel + i0]
    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int b_out = bx;
    int m_out = by;
    int c_in = tz;


    int Width_grid = (Width_out - 1) / TILE_WIDTH + 1;
    int w = TILE_WIDTH * (bz % Width_grid) + tx;
    int h = TILE_WIDTH * (bz / Width_grid) + ty;

    int heightBoundary = TILE_WIDTH * (bz / Width_grid);
    int widthBoundary = TILE_WIDTH * (bz % Width_grid);

    if (h < Height_out && w < Width_out) {
        float result = 0;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++)
            {
                result += in_4d(b_out, c_in, h + p, w + q) * mask_4d(m_out, c_in, p, q);
            }
        }
        sharedMemTree(ty, tx, c_in) = result;

        for (int stride = 1; stride < Channel; stride *= 2) {
            __syncthreads();
            if ((tz % (2 * stride) == 0) && (tz + stride < Channel)) {
                sharedMemTree(ty, tx, tz) += sharedMemTree(ty, tx, tz + stride);
            }
        }
        __syncthreads();
        if (tz == 0) {
            out_4d(b_out, m_out, h, w) = sharedMemTree(ty, tx, 0);
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef sharedMemTree
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void**)device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mask, host_mask, (Map_out * Channel * K * K) * sizeof(float));
    //cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, Channel);
    dim3 gridDim(Batch, Map_out, ceil((float)(Height - K + 1) / TILE_WIDTH) * ceil((float)(Width - K + 1) / TILE_WIDTH));
    conv_forward_kernel<<<gridDim, blockDim, Channel * TILE_WIDTH * TILE_WIDTH * sizeof(float)>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}




