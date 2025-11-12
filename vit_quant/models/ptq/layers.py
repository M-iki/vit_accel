# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from .bit_type import BIT_TYPE_DICT, BIT_TYPE_LIST
from .observer import build_observer
from .quantizer import build_quantizer
# from models import BIT_TYPE_DICT
from .observer import utils

class QConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'conv_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x, bit_config):

        if self.calibrate:
            for bit_type in BIT_TYPE_LIST:
                if bit_type == BIT_TYPE_DICT['uint8']:
                    continue
                # FIXME:
                self.quantizer.bit_type = bit_type
                self.observer.bit_type = bit_type
                if bit_type == BIT_TYPE_DICT['int8']:
                    # TODO:
                    self.observer.calibration_mode = 'layer_wise'
                else: 
                    self.observer.calibration_mode = 'channel_wise'
                self.quantizer.observer.update(self.weight)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x, others=[self.bias,self.stride,self.padding,self.dilation,self.groups])
        if not self.quant:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        if bit_config:
            bit_type = 'int' + str(bit_config)
            self.quantizer.bit_type = BIT_TYPE_DICT[bit_type]
            self.observer.bit_type = BIT_TYPE_DICT[bit_type]
        weight = self.quantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
        # y = F.conv2d(
        #         x[0],
        #         self.weight,
        #         self.bias,
        #         self.stride,
        #         self.padding,
        #         self.dilation,
        #         self.groups,
        #     )
        # weight = self.quantizer(self.weight)
        # y_q = F.conv2d(x[1], weight, self.bias, self.stride, self.padding,
        #                 self.dilation, self.groups)
        # return [y, y_q]


class QLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'linear_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x, global_distance=[], bit_config=None, weight_smoothed=None, attn=False, attn_para=None):
        # if self.calibrate:
        #     self.quantizer.observer.update(self.weight)
        #     if self.last_calibrate:
        #         self.quantizer.update_quantization_params(x, others=[self.bias])
        # if not self.quant:
        #     return F.linear(x, self.weight, self.bias)
        # weight = self.quantizer(self.weight)
        # return F.linear(x, weight, self.bias)
        if weight_smoothed==None:
            weight_smoothed = self.weight
        if not self.quant:
            # y = F.linear(x, self.weight, self.bias)
            y = F.linear(x, weight_smoothed, self.bias)

        if self.calibrate:
            # if self.last_calibrate: 
            distance = []
            for bit_type in BIT_TYPE_LIST:
                if bit_type == BIT_TYPE_DICT['uint8']:
                    continue
                # FIXME:
                self.quantizer.bit_type = bit_type
                self.observer.bit_type = bit_type
                if bit_type == BIT_TYPE_DICT['int8']:
                    # TODO:
                    self.observer.calibration_mode = 'layer_wise'
                else: 
                    self.observer.calibration_mode = 'channel_wise'
                
                self.quantizer.observer.update(weight_smoothed)
                self.quantizer.update_quantization_params(x, others=[self.bias], attn=attn, attn_para=attn_para)
                weight = self.quantizer(weight_smoothed)
                # y_q = F.linear(x, weight, self.bias)
                # TODO:
                # distance.append(utils.lp_loss(y, y_q, p=2.0, reduction='all'))
                distance.append(utils.lp_loss(weight_smoothed, weight, p=2.0, reduction='all'))
            global_distance.append(distance)
        if not self.quant:
            return y
        if bit_config:
            bit_type = 'int' + str(bit_config)
            self.quantizer.bit_type = BIT_TYPE_DICT[bit_type]
            self.observer.bit_type = BIT_TYPE_DICT[bit_type]
        weight = self.quantizer(weight_smoothed)
        return F.linear(x, weight, self.bias)


class QAct(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x, asymmetric=False, attn=False, attn_para=None):
        if self.calibrate: 
            if asymmetric:
                self.quantizer.bit_type = BIT_TYPE_DICT['uint8']
                self.observer.bit_type = BIT_TYPE_DICT['uint8']
                self.observer.symmetric = False
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                # import ipdb;ipdb.set_trace()
                self.quantizer.update_quantization_params(x, attn=attn, attn_para=attn_para)
        if not self.quant:
            return x
        x = self.quantizer(x)
        return x
        # y = x[0]
        # y_q = self.quantizer(x[1])
        # return [y, y_q]


class QIntLayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(QIntLayerNorm, self).__init__(normalized_shape, eps,
                                            elementwise_affine)
        assert isinstance(normalized_shape, int)
        self.mode = 'ln'

    def get_MN(self, x):
        bit = 7
        N = torch.clamp(bit - torch.floor(torch.log2(x)), 0, 31)
        M = torch.clamp(torch.floor(x * torch.pow(2, N)), 0, 2**(bit + 1) - 1)
        return M, N

    def forward(self,
                x,
                in_quantizer=None,
                out_quantizer=None,
                out_quantizer_scale=None,
                in_scale_expand=1):
        
        # y = F.layer_norm(x[0], self.normalized_shape, self.weight, self.bias,
        #                      self.eps)
        # if self.mode == 'ln':
        #     y_q = F.layer_norm(x[1], self.normalized_shape, self.weight, self.bias,
        #                      self.eps)
        if self.mode == 'ln':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif self.mode == 'int':
            in_scale = in_quantizer.scale
            if in_scale_expand != 1:
                in_scale = in_scale.unsqueeze(-1).expand(
                    -1, in_scale_expand).T.reshape(-1)
            out_scale_global = out_quantizer.scale
            assert in_scale is not None and out_scale_global is not None
            channel_nums = x.shape[-1]
            in_scale = in_scale.reshape(1, 1, -1)
            if out_quantizer_scale != None:
                out_scale = out_scale_global*out_quantizer_scale 
            else:
                out_scale = out_scale_global
            out_scale = out_scale.reshape(1, 1, -1)
            x_q = (x / in_scale).round()
            in_scale1 = in_scale.min()
            in_scale_mask = (in_scale / in_scale1).round()

            x_q = x_q * in_scale_mask

            mean_x_q = x_q.mean(dim=-1) * in_scale1
            std_x_q = (in_scale1 / channel_nums) * torch.sqrt(
                channel_nums * (x_q**2).sum(dim=-1) - x_q.sum(dim=-1)**2)

            A = (in_scale1 / std_x_q).unsqueeze(-1) * \
                self.weight.reshape(1, 1, -1) / out_scale
            A_sign = A.sign()
            M, N = self.get_MN(A.abs())
            B = ((self.bias.reshape(1, 1, -1) -
                    (mean_x_q / std_x_q).unsqueeze(-1) *
                    self.weight.reshape(1, 1, -1)) / out_scale *
                    torch.pow(2, N)).round()

            x_q = ((A_sign * M * x_q + B) / torch.pow(2, N)).round()
            return x_q * out_scale
        else:
            raise NotImplementedError
        # return [y, y_q]


class QIntSoftmax(nn.Module):

    def __init__(self,
                 log_i_softmax=False,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QIntSoftmax, self).__init__()

        self.log_i_softmax = log_i_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor**2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            # TODO:
            # n =  30 # sufficiently large integer
            n =  32 # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor

        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scale):
        if self.log_i_softmax and scale is not None:
            exp_int, exp_int_sum = self.int_softmax(x, scale)
            softmax_out = torch.round(exp_int_sum / exp_int)
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2**self.bit_type.bits
            qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
            deq_softmax = 2**(-qlog)
            deq_softmax[mask] = 0
            return deq_softmax
        else:
            # x = x.softmax(dim=-1)
            # if self.calibrate:
            #     self.quantizer.observer.update(x)
            #     if self.last_calibrate:
            #         self.quantizer.update_quantization_params(x)
            # if not self.quant:
            #     return x
            # x = self.quantizer(x)
            # return x
            x = x.softmax(dim=-1)
            # if self.calibrate:
            #     self.quantizer.observer.update(x)
            #     if self.last_calibrate:
            #         self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            # x = self.quantizer(x)
            return x

        # y = x[0].softmax(dim=-1)
        # # if self.log_i_softmax:
        # exp_int, exp_int_sum = self.int_softmax(x[1], scale)
        # softmax_out = torch.round(exp_int_sum / exp_int)
        # rounds = self.log_round(softmax_out)
        # mask = rounds >= 2**self.bit_type.bits
        # qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
        # deq_softmax = 2**(-qlog)
        # deq_softmax[mask] = 0
        # y_q = deq_softmax
        # # else:
        # #     y_q = x[1].softmax(dim=-1)
        # if self.calibrate:
        #     self.quantizer.observer.update(y)
        #     if self.last_calibrate:
        #         self.quantizer.update_quantization_params([y, y_q])
        
        # return [y, y_q]
        
        
class QIntLinearSoftmax(nn.Module):

    def __init__(self,
                 log_i_softmax=False,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QIntLinearSoftmax, self).__init__()

        self.log_i_softmax = log_i_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor**2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            # TODO:
            # n =  30 # sufficiently large integer
            n =  32 # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor

        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        # exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        x_int_sum = x_int.sum(dim=-1, keepdim=True)
        return x_int, x_int_sum

    def forward(self, x, scale):
        if self.log_i_softmax and scale is not None:
            exp_int, exp_int_sum = self.int_softmax(x, scale)
            softmax_out = torch.round(exp_int_sum / exp_int)
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2**self.bit_type.bits
            qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
            deq_softmax = 2**(-qlog)
            deq_softmax[mask] = 0
            return deq_softmax
        else:
            # x = x.softmax(dim=-1)
            # if self.calibrate:
            #     self.quantizer.observer.update(x)
            #     if self.last_calibrate:
            #         self.quantizer.update_quantization_params(x)
            # if not self.quant:
            #     return x
            # x = self.quantizer(x)
            # return x
            x = x.softmax(dim=-1)
            # if self.calibrate:
            #     self.quantizer.observer.update(x)
            #     if self.last_calibrate:
            #         self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            # x = self.quantizer(x)
            return x

        # y = x[0].softmax(dim=-1)
        # # if self.log_i_softmax:
        # exp_int, exp_int_sum = self.int_softmax(x[1], scale)
        # softmax_out = torch.round(exp_int_sum / exp_int)
        # rounds = self.log_round(softmax_out)
        # mask = rounds >= 2**self.bit_type.bits
        # qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
        # deq_softmax = 2**(-qlog)
        # deq_softmax[mask] = 0
        # y_q = deq_softmax
        # # else:
        # #     y_q = x[1].softmax(dim=-1)
        # if self.calibrate:
        #     self.quantizer.observer.update(y)
        #     if self.last_calibrate:
        #         self.quantizer.update_quantization_params([y, y_q])
        
        # return [y, y_q]
