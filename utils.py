import pynvml

def print_memory_usage():
    pynvml.nvmlInit()  # Initialize NVML
    device_count = pynvml.nvmlDeviceGetCount()  # Get the number of GPUs

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # Get the handle for the GPU
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # Get memory info
        name = pynvml.nvmlDeviceGetName(handle)  # Get GPU name

        print(f"GPU {i}: {name.decode('utf-8')}")
        print(f"  Total Memory: {info.total / (1024 ** 2):.2f} MiB")
        print(f"  Free Memory: {info.free / (1024 ** 2):.2f} MiB")
        print(f"  Used Memory: {info.used / (1024 ** 2):.2f} MiB")

    pynvml.nvmlShutdown()  # Shutdown NVML

if __name__ == '__main__':
    print_memory_usage()