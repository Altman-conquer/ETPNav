from time import sleep

import torch


def request(i):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{i}")
        try:
            # 在设备上申请一小段显存 (例如 1MB)
            tensor = torch.empty((256, 256), device=device)
            print(f"成功在 GPU {i} 上申请显存")
        except RuntimeError as e:
            print(f"无法在 GPU {i} 上申请显存: {e}")
    else:
        print("CUDA 不可用")

for i in range(8):
    device = torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")
    print("GPU 编号: {}".format(device))
    print("GPU 名称: {}".format(torch.cuda.get_device_name(i)))

    request(i)
    sleep(5)  # 等待一秒钟以便观察输出
