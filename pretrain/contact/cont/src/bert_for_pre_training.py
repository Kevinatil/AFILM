# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Bert for pretraining."""
import numpy as np

import mindspore.nn as nn
from mindspore.common.initializer import initializer, TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore import context
from .bert_model import BertModel

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class GetContOutput(nn.Cell): # mutated position in germline

    def __init__(self, config):
        super(GetContOutput, self).__init__()
        self.width = config.hidden_size
        self.reshape = P.Reshape()
        self.gather = P.Gather()
        self.scale = np.sqrt(self.width)

        weight_init = TruncatedNormal(config.initializer_range)
        self.linear = nn.Dense(self.width,
                               self.width,
                               weight_init=weight_init,
                               activation=config.hidden_act).to_float(config.compute_type)
        self.bias = Parameter(Tensor([0], dtype=mstype.float32))

        #self.log_softmax = nn.LogSoftmax(axis=-1)
        self.softmax = nn.Softmax()
        self.shape_flat_offsets = (-1, 1)
        self.last_idx = (-1,)
        self.shape_flat_sequence_tensor = (-1, self.width)
        self.cast = P.Cast()
        self.compute_type = config.compute_type
        self.dtype = config.dtype
        self.transpose = P.Transpose()
        self.sigmoid = nn.Sigmoid()
        self.bmm = P.BatchMatMul()
        self.slice = P.StridedSlice()

    def construct(self,
                  input_tensor
                 ):
        """Get output log_probs"""
        input_tensor=input_tensor[:,2:112,:]
        #batch_size = input_tensor.shape[0] #P.Shape()(input_tensor)[0]
        #input_tensor = self.slice(input_tensor,
        #                            (0, 2, 0),
        #                            (int(batch_size), 152, self.width),
        #                            (1, 1, 1))
        
        input_shape = P.Shape()(input_tensor) # [B, L, D]
        
        input_tensor_flat = self.reshape(input_tensor, self.shape_flat_sequence_tensor) # [BL, D]
        input_tensor_flat = self.cast(input_tensor_flat, self.compute_type)
        input_tensor = self.cast(input_tensor, self.compute_type)

        input_tensor_flat = self.linear(input_tensor_flat)#.view(input_shape[0], input_shape[1], -1)
        input_tensor_flat = self.reshape(input_tensor_flat, (input_shape[0], input_shape[1], -1))
        input_tensor = self.bmm(input_tensor_flat, self.transpose(input_tensor,(0,2,1)))/self.scale + self.bias
        input_tensor = self.reshape(input_tensor, (input_shape[0], -1))
        input_tensor = self.softmax(input_tensor)
        input_tensor = self.cast(input_tensor, self.dtype)
        
        return input_tensor

    
class BertPreTraining(nn.Cell):
    """
    Bert pretraining network.

    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings.

    Returns:
        Tensor, prediction_scores, seq_relationship_score.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings):
        super(BertPreTraining, self).__init__()
        self.encoder = BertModel(config, is_training, use_one_hot_embeddings)
        
        self.cls_cont = GetContOutput(config)
        
        #self.squeeze_1 = P.Squeeze(axis=1)
        #self.slice = P.StridedSlice()
        #self.hidden_size=config.hidden_size
        #self.cls_anc = GetAncestorOutput(config)
        
        #self.cls_mut_pos = GetMutPositionOutput(config)
        #self.cls_mut_mut = GetMutMutationOutput(config)

    def construct(self, input_ids_mlm, input_mask_mlm, token_type_ids,
                  ):
        
        sequence_output, embedding_table = self.encoder(input_ids_mlm, input_mask_mlm, token_type_ids,)
        prediction_scores_cont = self.cls_cont(sequence_output)
        
        return prediction_scores_cont #, prediction_scores_anc, prediction_scores_mut_mut, prediction_scores_mut_pos


class BertPretrainingLoss(nn.Cell):
    """
    Provide bert pre-training loss.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        Tensor, total loss.
    """

    def __init__(self, config):
        super(BertPretrainingLoss, self).__init__()
        self.vocab_size = config.vocab_size
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.slice = P.StridedSlice()

    def construct(self, prediction_scores_cont, cont_label, cont_label_weight, seq_lens,
                  #prediction_scores_anc, ancestor_label,
                  #prediction_scores_mut_mut, mutation_masked_lm_ids, mutation_masked_lm_weights,
                  #prediction_scores_mut_pos, mutation_germline_mask, mutation_germline_mutation
                  ):
        """Defines the computation performed."""
        
        cont_label = self.reshape(cont_label, self.last_idx) # [BLL, ]
        prediction_scores_cont = self.reshape(prediction_scores_cont, self.last_idx) # [BLL, ]

        tp_=self.cast(cont_label * prediction_scores_cont, mstype.float32)
        cont_loss = self.reduce_mean(tp_, ()) #self.reduce_mean(label_ * pred_, ())
        
        
        
        #cont_label_weight = self.cast(self.reshape(cont_label_weight, self.last_idx), mstype.float32)
        #one_hot_labels = self.onehot(cont_label, 2, self.on_value, self.off_value)
        #per_example_loss = self.neg(self.reduce_sum(prediction_scores_mut_pos * one_hot_labels, self.last_idx))
        #numerator = self.reduce_sum(cont_label_weight * per_example_loss, ())
        #denominator = self.reduce_sum(cont_label_weight, ()) + self.cast(F.tuple_to_array((1e-5,)), mstype.float32)
        ##print(mutation_germline_mask.sum(), denominator)
        #mut_pos_loss = numerator / denominator
        
        #print('loss: ',masked_lm_loss, anc_loss, mut_mut_loss, mut_pos_loss)

        return cont_loss #+ anc_loss + mut_mut_loss + mut_pos_loss


class BertNetworkWithLoss(nn.Cell):
    """
    Provide bert pre-training loss through network.

    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(BertNetworkWithLoss, self).__init__()
        self.predictor = BertPreTraining(config, is_training, use_one_hot_embeddings)
        self.loss = BertPretrainingLoss(config)
        
        self.cast = P.Cast()

    def construct(self,
                input_ids,
                input_mask,
                token_type_ids,
                cont_label,
                cont_label_weight,
                seq_len,
                 ):
        """Get pre-training loss"""
        # prediction_scores_mlm, prediction_scores_anc, \
        # prediction_scores_mut_mut, prediction_scores_mut_pos = self.predictor(input_ids, 
        #                                 input_mask, masked_lm_positions, 
        #                                 ancestor_ids, ancestor_mask, 
        #                                 mutation_input_ids, mutation_input_mask, mutation_masked_lm_positions)

        # loss_all = self.loss(prediction_scores_mlm, masked_lm_ids, masked_lm_weights, 
        #                      prediction_scores_anc, ancestor_label,
        #                      prediction_scores_mut_mut, mutation_masked_lm_ids, mutation_masked_lm_weights,
        #                      prediction_scores_mut_pos, mutation_germline_mask, mutation_germline_mutation
        #                     )


        prediction_scores_cont = self.predictor(input_ids, input_mask, token_type_ids,)
        
        loss_all = self.loss(prediction_scores_cont, cont_label, cont_label_weight, seq_len,)
        
        return self.cast(loss_all, mstype.float32)



grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class BertTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self,
                input_ids,
                input_mask,
                token_type_ids,
                cont_label,
                cont_label_weight,
                seq_len,

                sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_ids,
                            cont_label,
                            cont_label_weight,
                            seq_len,
                           )
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(
                                                input_ids,
                                                input_mask,
                                                token_type_ids,
                                                cont_label,
                                                cont_label_weight,
                                                seq_len,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        degree_sens = self.cast(scaling_sens * self.degree, mstype.float32)
        grads = self.hyper_map(F.partial(grad_scale, degree_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond, scaling_sens)



class BertPretrainEval(nn.Cell):
    '''
    Evaluate MaskedLM prediction scores
    '''
    def __init__(self, config, network=None):
        super(BertPretrainEval, self).__init__(auto_prefix=False)
        if network is None:
            self.network = BertPreTraining(config, False, False)
        else:
            self.network = network
        self.argmax = P.Argmax(axis=-1, output_type=mstype.int32)
        self.equal = P.Equal()
        self.sum = P.ReduceSum()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.allreduce = P.AllReduce()
        self.reduce_flag = False
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reduce_flag = True

    def construct(self,
                  input_ids,
                  input_mask,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  
                  token_type_ids,
                  ):
        """Calculate prediction scores"""
        bs, _ = self.shape(input_ids)
        mlm = self.network(
            input_ids, 
            input_mask, masked_lm_positions, token_type_ids,
        )
        index = self.argmax(mlm)
        index = self.reshape(index, (bs, -1))
        eval_acc = self.equal(index, masked_lm_ids)
        eval_acc = self.cast(eval_acc, mstype.float32)
        real_acc = eval_acc * masked_lm_weights
        acc = self.sum(real_acc)
        total = self.sum(masked_lm_weights)

        if self.reduce_flag:
            acc = self.allreduce(acc)
            total = self.allreduce(total)

        return acc, total

