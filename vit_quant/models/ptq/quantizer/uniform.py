# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from ..bit_type import BIT_TYPE_DICT
from .base import BaseQuantizer


class UniformQuantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type)
        # self.int4_scale = None
        # self.int4_zero_point = None
        # self.int6_scale = None
        # self.int6_zero_point = None
        # self.int8_scale = None
        # self.int8_zero_point = None
        # self.int10_scale = None
        # self.int10_zero_point = None
        self.scale = None
        self.zero_point = None
        self.dic_scale = {}
        self.dic_zero_point = {}

    def update_quantization_params(self, *args, **kwargs):
        scale, zero_point = self.observer.get_quantization_params(
            *args, **kwargs)
        if self.module_type == 'activation':
            self.scale = scale
            self.zero_point = zero_point
        # elif self.bit_type == BIT_TYPE_DICT['int4']:
        #     self.int4_scale = scale
        #     self.int4_zero_point = zero_point
        # elif self.bit_type == BIT_TYPE_DICT['int6']:
        #     self.int6_scale = scale
        #     self.int6_zero_point = zero_point
        # elif self.bit_type == BIT_TYPE_DICT['int8']:
        #     self.int8_scale = scale
        #     self.int8_zero_point = zero_point
        # elif self.bit_type == BIT_TYPE_DICT['int10']:
        #     self.int10_scale = scale
        #     self.int10_zero_point = zero_point
        # else:
        #     AssertionError
        else:
            self.dic_scale[self.bit_type.name] = scale
            self.dic_zero_point[self.bit_type.name] = zero_point
            

    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            if self.module_type == 'activation':
                scale = self.scale
            # elif self.bit_type == BIT_TYPE_DICT['int4']:
            #     scale = self.int4_scale
            # elif self.bit_type == BIT_TYPE_DICT['int6']:
            #     scale = self.int6_scale
            # elif self.bit_type == BIT_TYPE_DICT['int8']:
            #     scale = self.int8_scale
            # elif self.bit_type == BIT_TYPE_DICT['int10']:
            #     scale = self.int10_scale
            # else:
            #     AssertionError
            else:
                scale = self.dic_scale[self.bit_type.name]
        if zero_point is None:
            if self.module_type == 'activation':
                zero_point = self.zero_point
            # elif self.bit_type == BIT_TYPE_DICT['int4']:
            #     zero_point = self.int4_zero_point
            # elif self.bit_type == BIT_TYPE_DICT['int6']:
            #     zero_point = self.int6_zero_point
            # elif self.bit_type == BIT_TYPE_DICT['int8']:
            #     zero_point = self.int8_zero_point
            # elif self.bit_type == BIT_TYPE_DICT['int10']:
            #     zero_point = self.int10_zero_point
            # else:
            #     AssertionError
            else:
                zero_point = self.dic_zero_point[self.bit_type.name]
            # zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs.cuda() / scale.cuda() + zero_point.cuda()
        outputs = outputs.round().clamp(self.bit_type.lower_bound,
                                        self.bit_type.upper_bound)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        # if scale is None:
        #     scale = self.scale
        # if zero_point is None:
        #     zero_point = self.zero_point
        if scale is None:
            if self.module_type == 'activation':
                scale = self.scale
            # elif self.bit_type == BIT_TYPE_DICT['int4']:
            #     scale = self.int4_scale
            # elif self.bit_type == BIT_TYPE_DICT['int6']:
            #     scale = self.int6_scale
            # elif self.bit_type == BIT_TYPE_DICT['int8']:
            #     scale = self.int8_scale
            # elif self.bit_type == BIT_TYPE_DICT['int10']:
            #     scale = self.int10_scale
            else:
                # AssertionError
                scale = self.dic_scale[self.bit_type.name]
        if zero_point is None:
            if self.module_type == 'activation':
                zero_point = self.zero_point
            # elif self.bit_type == BIT_TYPE_DICT['int4']:
            #     zero_point = self.int4_zero_point
            # elif self.bit_type == BIT_TYPE_DICT['int6']:
            #     zero_point = self.int6_zero_point
            # elif self.bit_type == BIT_TYPE_DICT['int8']:
            #     zero_point = self.int8_zero_point
            # elif self.bit_type == BIT_TYPE_DICT['int10']:
            #     zero_point = self.int10_zero_point
            else:
                # AssertionError
                zero_point = self.dic_zero_point[self.bit_type.name]
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs.cuda() - zero_point.cuda()) * scale.cuda()
        return outputs
