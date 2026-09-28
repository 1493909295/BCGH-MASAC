# import torch
# flag = torch.cuda.is_available()
# if flag:
#     print("CUDA可使用")
# else:
#     print("CUDA不可用")
#
# ngpu= 1
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# print("驱动为：",device)
# print("GPU型号： ",torch.cuda.get_device_name(0))


import torch
import xuance
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
