import torch

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        print(f"Device {i}: {props.name}")
        print(f"  Free memory: {free_mem / 1e9:.2f} GB")
        print(f"  Total memory: {total_mem / 1e9:.2f} GB")
else:
    print("CUDA is not available")