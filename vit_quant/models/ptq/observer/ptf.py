# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseObserver
from .utils import lp_loss


class PtfObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver, self).__init__(module_type, bit_type,
                                          calibration_mode)

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


    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val
        self.inputs = inputs

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        # TODO:
        # ############## asymmetric ################
        # max_val_t = max_val.max()
        # min_val_t = min_val.min()
        # scale8 = (max_val_t - min_val_t) / float(qmax - qmin)
        ################# symmetric #################
        max_val_t = torch.max(-min_val.min(), max_val.max())
        scale8 = torch.ones_like(max_val.max(), dtype=torch.float32)
        scale8 = 2*max_val_t / float(qmax - qmin)
        # # # TODO: ######### round 2^n #########
        def round_ln(x, type=None):
            if type == 'ceil':
                return torch.ceil(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            elif type == 'floor':
                return torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            else:
                y = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
                out = torch.gt((x-2**y),(2**(y+1)-x))
                return out+y

        def round_x(scale, zero_point=False):
            alpha_round = round_ln(scale, 'round')
            alpha_floor = round_ln(scale, 'floor')
            alpha = alpha_round
            if not zero_point:
                zero_point = torch.Tensor([0]).cuda()
            # print(scale.shape)
            dim = 1
            for j in range(dim):
                # if dim == 1:
                #     weight = x
                # else:
                #     weight = x[j,:].unsqueeze(-2)
                # FIXME:
                weight = self.inputs
                # print(self.inputs[0]==self.v)
                weight_1 = ((weight / 2**alpha_floor[j] + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**alpha_floor[j]
                weight_2 = ((weight / 2**(alpha_floor[j]+1) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j]+1)
                score1 = lp_loss(self.inputs, weight_1, p=2.0, reduction='all')
                score2 = lp_loss(self.inputs, weight_2, p=2.0, reduction='all')
                score = [score1, score2]
                if score.index(min(score)) == 0:
                    alpha[j] = alpha_floor[j]
                else:
                    alpha[j] = alpha_floor[j]+1
            return alpha
        # FIXME:
        # alpha_ceil = round_ln(scale8, 'ceil')
        # alpha_x = round_x(scale8)
        # scale8 = 2**alpha_ceil
        #############################################
        scale8.clamp_(self.eps)
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        # scale0 = scale1 / 2
        # TODO:
        # ############## asymmetric ################
        # zero_point = qmin - torch.round(min_val_t / scale8)
        # zero_point.clamp_(qmin, qmax)
        ################# symmetric #################
        zero_point = torch.zeros_like(max_val.max(), dtype=torch.int64)
        #############################################

        self.scale_mask = torch.ones_like(max_val)
        for j in range(inputs.shape[2]):
            # FIXME:
            data = inputs[..., j].unsqueeze(-1)
            data_gt = inputs[..., j].unsqueeze(-1)
            # data_q0 = ((data / scale0 + zero_point).round().clamp(qmin, qmax) -
            #            zero_point) * scale0
            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            # score0 = lp_loss(data_gt, data_q0, p=2.0, reduction='all')
            score1 = lp_loss(data_gt, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data_gt, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data_gt, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data_gt, data_q8, p=2.0, reduction='all')
            # score = [score0, score1, score2, score4, score8]
            score = [score1, score2, score4, score8]
            self.scale_mask[j] *= 2**score.index(min(score))
        # TODO:
        scale = scale1 * self.scale_mask
        return scale, zero_point

