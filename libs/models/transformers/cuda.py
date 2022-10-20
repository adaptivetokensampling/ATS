import time
from collections import namedtuple
from string import Template

import cupy
import torch
import math
# HELPER FUNCTIONS

Stream = namedtuple('Stream', ['ptr'])

ROUNDING = 8
CUDA_NUM_THREADS = 256
CHANNELS_PER_THREAD = 2

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

@cupy.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code,options=('--restrict','--use_fast_math'))
    return kernel_code.get_function(kernel_name)

def GET_BLOCKS(N, NTHREADS):
    return min((N + NTHREADS - 1) // (NTHREADS), 256*256-1)

def cudaok(x):
    return x.is_cuda and x.is_contiguous()

def ctime():
    torch.cuda.synchronize()
    return time.perf_counter()



_gather_tokens_kernel = '''
extern "C"
__global__ void gather_tokens_kernel(const float* __restrict__  const data_in, float* __restrict__ const data_out, const int* __restrict__  const active_positions_list, const int n) {

#define BATCHSIZE ${batchsize}
#define CHANNELS ${channels}
#define LENGTH ${length}

for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += blockDim.y * gridDim.y){
    const int k = (int) active_positions_list[i];
    const int b = k / (LENGTH);
    const int pos = k % (LENGTH);
    const int offset = b * CHANNELS * LENGTH + pos;  
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; \
        c < CHANNELS; c += blockDim.x * gridDim.x){
        data_out[i*CHANNELS+c] = data_in[offset+c*LENGTH];
    }
}
}
'''

def gather_tokens(data_in, active_positions_list, divisible=ROUNDING):
    with torch.cuda.device_of(data_in):
        batchsize, channels, length = data_in.shape

        assert cudaok(data_in)
        assert cudaok(active_positions_list)
        npixels = len(active_positions_list)
#         n_out = (npixels//divisible + 1)*divisible if divisible > 0 else npixels
        n_out = npixels
        data_out = torch.empty((n_out, channels, 1), device='cuda')
        data_out[npixels:] = 0

        threadsx =  min(math.ceil(channels/CHANNELS_PER_THREAD), CUDA_NUM_THREADS) 
        threadsy = max(CUDA_NUM_THREADS//threadsx, 1)
        block = (threadsx, threadsy,1)
        grid = (1,GET_BLOCKS(npixels, threadsy),1)

        f = load_kernel('gather_tokens_kernel', _gather_tokens_kernel, 
            batchsize=batchsize, channels=channels,
            length=length)
        f(block=block, grid=grid,
            args=[data_in.data_ptr(), data_out.data_ptr(), active_positions_list.data_ptr(), int(npixels)],
            stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return data_out


_scatter_tokens_kernel = '''
extern "C"
__global__ void scatter_tokens_kernel(const float* __restrict__  const data_in, float* __restrict__ const data_out, const int* __restrict__  const active_positions_list, const int n) {

#define BATCHSIZE ${batchsize}
#define CHANNELS ${channels}
#define LENGTH ${length}
#define OUTPUT_CHANNEL_OFFSET ${output_channel_offset}

for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += blockDim.y * gridDim.y){
    const int k = (int) active_positions_list[i];
    const int b = k / (LENGTH);
    const int pos = k % (LENGTH);
    const int offset = b * CHANNELS * LENGTH + pos;  
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; \
        c < CHANNELS; c += blockDim.x * gridDim.x){
        if(${sum_result} > 0){
            atomicAdd(data_out+offset+c*LENGTH, data_in[i*CHANNELS+c]);          
        } else {
            data_out[offset+(OUTPUT_CHANNEL_OFFSET+c)*LENGTH] = data_in[i*CHANNELS+c];
        }
    }
}
}
'''

def scatter_tokens(data_in, data_out, active_positions_list, sum_out=False, output_channel_offset=0):
    batchsize, channels, length = data_out.shape

    assert cudaok(data_in)
    assert cudaok(data_out)
    assert cudaok(active_positions_list)
    assert len(active_positions_list) <= data_in.shape[0], (len(active_positions_list), data_in.shape)
    assert data_in.shape[1] == (data_out.shape[1] - output_channel_offset)
#     assert data_in.shape[2] == 1
#     assert data_in.shape[3] == 1

    threadsx =  min(math.ceil(channels/CHANNELS_PER_THREAD), CUDA_NUM_THREADS) 
    threadsy = max(CUDA_NUM_THREADS//threadsx, 1)
    block = (threadsx, threadsy,1)
    grid = (1,GET_BLOCKS(len(active_positions_list), threadsy),1)

    with torch.cuda.device_of(data_in):
        f = load_kernel('scatter_tokens_kernel', _scatter_tokens_kernel, 
            batchsize=batchsize, channels=channels,
            length=length, sum_result=int(sum_out), output_channel_offset=output_channel_offset)
        f(block=block, grid=grid,
            args=[data_in.data_ptr(), data_out.data_ptr(), active_positions_list.data_ptr(), int(len(active_positions_list))],
            stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return data_out
