import torch

def get_least_used_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return torch.device("cpu")

    free_mem = []
    for i in range(num_gpus):
        stats = torch.cuda.mem_get_info(i)  # (free, total)
        free_mem.append(stats[0])
    best_gpu = int(torch.argmax(torch.tensor(free_mem)))
    print(f"@hardware-mgmt: Using GPU {best_gpu} (free memory: {free_mem[best_gpu]/1e9:.2f} GB)")
    return f"cuda:{best_gpu}"

# import pynvml

# def get_least_used_gpu():
#     pynvml.nvmlInit()
#     device_count = pynvml.nvmlDeviceGetCount()
#     free_mem = []
#     for i in range(device_count):
#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#         mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         free_mem.append(mem_info.free)
#     best_gpu = int(max(range(device_count), key=lambda i: free_mem[i]))
#     print(f"== Using GPU {best_gpu} (free memory: {free_mem[best_gpu]/1e9:.2f} GB)")
#     pynvml.nvmlShutdown()
#     return f"cuda:{best_gpu}"

# if __name__ == '__main__':
#     print(get_least_used_gpu())