import torch
import numpy as np
scale_global = torch.Tensor([1])
# channel num = 3
# alpha = 1, 2
scale_original = torch.Tensor([2.92, 3.13])
ratio = torch.ones_like(scale_original)
ratio = scale_original / scale_global
# print(torch.div(torch.log(torch.Tensor([2.92, 3.13])),torch.log(torch.Tensor([2]))))
# print(torch.log(torch.Tensor([2])))
def round_ln(x):
    y = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]))))
    print((torch.div(torch.log(x),torch.log(torch.Tensor([2]))).round()))
    out = torch.gt((x-2**y),(2**(y+1)-x))
    print(out+y)
    #     return (y+1)
    # else:
    #     return x
    return y

alpha = round_ln(ratio)
scale = scale_original*2**alpha
