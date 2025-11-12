# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseObserver
from .utils import lp_loss
from torch.nn import functional as F


class MinmaxObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(MinmaxObserver, self).__init__(module_type, bit_type,
                                             calibration_mode)
        self.symmetric = self.bit_type.signed

    def update(self, v):
        self.v = v
        v = self.reshape_tensor(v)
        # self.v = v
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()


    def get_quantization_params(self, x, others=None, attn=False, attn_para=None, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val
        self.input = x
        self.others = others
        self.attn = attn
        self.attn_para = attn_para

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        def round_ln(x, type=None):
            if type == 'ceil':
                return torch.ceil(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            elif type == 'floor':
                return torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            else:
                y = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
                out = torch.gt((x-2**y),(2**(y+1)-x))
                return out+y
            # floor = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            # for j in range(self.v.shape[0]):
        
        def get_attn(x):
            B, N, M = x.shape 
            qkv = x.reshape(B, N, 3, self.attn_para[0],
            self.attn_para[1] // self.attn_para[0]).permute(2, 0, 3, 1, 4)
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )
            attn = (q @ k.transpose(-2, -1)) * self.attn_para[2]
            attn = attn.softmax(dim=-1)
            return (attn @ v).transpose(1, 2).reshape(B, N, self.attn_para[1])
        
        def get_out(x, j, quant=False):
            
            if self.calibration_mode == 'channel_wise':
                weight = self.v[j,...].unsqueeze(0)
                if self.others:
                    bias = self.others[0][j].unsqueeze(0)
            else:
                weight = self.v
                if self.others:
                    bias = self.others[0]
            # FIXME:
            # input_gt = self.input[0]
            # input_q = self.input[0]
            input_gt = self.input
            input_q = self.input
            if self.module_type == 'activation':
                if quant == True:
                    if self.attn and self.calibration_mode == 'layer_wise': 
                        return get_attn(x)
                    else:
                        return x
                else:
                    # return self.input[0]
                    if self.attn and self.calibration_mode == 'layer_wise': 
                        return get_attn(self.input)
                    else:
                        return self.input
            elif self.module_type == 'conv_weight':
                if quant == True:
                    return F.conv2d(input_q, x, bias, self.others[1], self.others[2], self.others[3], self.others[4])
                    # return x
                else:
                    return F.conv2d(input_gt, weight, bias, self.others[1], self.others[2], self.others[3], self.others[4])
                    # return weight
            elif self.module_type == 'linear_weight': 
                if quant == True:
                    out = F.linear(input_q, x, bias)
                    # return x
                else:
                    out = F.linear(input_gt, weight, bias)
                if self.attn and self.calibration_mode == 'layer_wise': 
                    return get_attn(out)
                else:
                    return out 

                    # return weight 

        def round_x(scale, x, zero_point=False):
            alpha_round = round_ln(scale, 'round').cuda()
            alpha_floor = round_ln(scale, 'floor').cuda()
            alpha = alpha_round
            if not zero_point:
                zero_point = torch.Tensor([0]).cuda()
            # print(scale.shape)
            if self.calibration_mode == 'layer_wise':
                dim = 1
            else:
                dim = scale.shape[0]
            for j in range(dim):
                if dim == 1:
                    weight = x.cuda()
                    if self.module_type == 'activation':
                        # FIXME:
                        # weight = self.input[0]
                        weight = self.input
                else:
                    weight = x[j,...].unsqueeze(0).cuda()
                weight_1 = ((weight / 2**(alpha_floor[j]-1) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j]-1)
                out_1 = get_out(weight_1, j, quant=True)
                # out_1 = weight_1
                weight_2 = ((weight / 2**(alpha_floor[j]) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j])
                out_2 = get_out(weight_2, j, quant=True)
                # TODO: expand the range
                weight_3 = ((weight / 2**(alpha_floor[j]+1) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j]+1)
                out_3 = get_out(weight_3, j, quant=True)
                weight_4 = ((weight / 2**(alpha_floor[j]+2) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j]+2)
                out_4 = get_out(weight_4, j, quant=True)
                # out_2 = weight_2
                out = get_out(weight, j, quant=False)
                # out = weight
                score1 = lp_loss(out, out_1, p=2.0, reduction='all')
                score2 = lp_loss(out, out_2, p=2.0, reduction='all')
                score3 = lp_loss(out, out_3, p=2.0, reduction='all')
                score4 = lp_loss(out, out_4, p=2.0, reduction='all')
                score = [score1, score2, score3, score4]
                # score = [score2, score3]
                indx = score.index(min(score))
                alpha[j] = alpha_floor[j] -1 + indx
                # alpha[j] = alpha_floor[j] + indx
            return alpha

        if self.symmetric:
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            # TODO: ########### 2^n ############
            alpha_x = round_x(scale, self.v)
            # alpha_x = round_ln(scale, self.v)
            scale = 2**alpha_x
            # if self.module_type in ['conv_weight', 'linear_weight']:
            #     # alpha_x = round_x(scale, self.v)
            #     # scale = 2**alpha_x
            #     pass
            # elif self.module_type == 'activation':
            #     alpha_x = round_x(scale, self.v)
            #     scale = 2**alpha_x
            #     # pass
            # ####################################
            scale.clamp_(self.eps)
        else:
            # zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            scale = (max_val - min_val) / float(qmax - qmin)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
            # TODO: ########### 2^n ############
            alpha_x = round_x(scale, self.v, zero_point)
            scale = 2**alpha_x
            ####################################
            scale.clamp_(self.eps)
        return scale, zero_point
