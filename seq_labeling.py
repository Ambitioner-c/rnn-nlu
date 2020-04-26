# -*- coding: utf-8 -*-
"""
Created on Sun Feb  28 11:32:21 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
# from six.moves import zip     # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example
from tensorflow.contrib.legacy_seq2seq import sequence_loss

from tensorflow.python.ops import rnn_cell_impl

linear = rnn_cell_impl._linear


def _step(time, sequence_length, min_sequence_length,
          max_sequence_length, zero_logit, generate_logit):
    # Step 1: 确定是否需要调用call_cell
    empty_update = lambda: zero_logit
    logit = control_flow_ops.cond(
        time < max_sequence_length, generate_logit, empty_update)

    # Step 2: 确定是否需要通过状态和/或输出进行复制
    existing_logit = lambda: logit

    def copy_through():
        # 使用广播选择来确定应该获取哪些值
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        copy_cond = (time >= sequence_length)
        return tf.where(copy_cond, zero_logit, logit)

    logit = control_flow_ops.cond(
        time < min_sequence_length, existing_logit, copy_through)
    logit.set_shape(zero_logit.get_shape())
    return logit


def attention_rnn(encoder_outputs,
                  encoder_state,
                  num_decoder_symbols,
                  sequence_length,
                  num_heads=1,
                  dtype=tf.float32,
                  use_attention=True,
                  loop_function=None,
                  scope=None):
    if use_attention:
        print('使用注意力RNN模型')
        if num_heads < 1:
            raise ValueError("少于1个头时，使用非注意力译码器。")

        with tf.variable_scope(scope or "attention_RNN"):
            output_size = encoder_outputs[0].get_shape()[1].value
            top_states = [tf.reshape(e, [-1, 1, output_size])
                          for e in encoder_outputs]
            attention_states = tf.concat(top_states, 1)
            if not attention_states.get_shape()[1:2].is_fully_defined():
                raise ValueError("Shape[1] 和 [2] 的注意力状态必须已知: %s"
                                 % attention_states.get_shape())

            # 需要重塑。
            batch_size = tf.shape(top_states[0])[0]
            attn_length = attention_states.get_shape()[1].value
            attn_size = attention_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            hidden = tf.reshape(
                attention_states, [-1, attn_length, 1, attn_size])
            hidden_features = []
            v = []
            # 注意力查询向量的大小。
            attention_vec_size = attn_size
            for a in xrange(num_heads):
                k = tf.get_variable("AttnW_%d" % a,
                                    [1, 1, attn_size, attention_vec_size])
                hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
                v.append(tf.get_variable("AttnV_%d" % a,
                                         [attention_vec_size]))

            def attention(query):
                """使用hidden_features和query将注意力蒙版放到hidden上。"""
                attn_weights = []
                # 注意阅读的结果将存储在这里。
                ds = []
                for j in xrange(num_heads):
                    with tf.variable_scope("Attention_%d" % j):
                        # y = linear(query, attention_vec_size, True)
                        y = linear(query, attention_vec_size, True)
                        y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                        # Attention mask is a softmax of v^T * tanh(...).
                        s = tf.reduce_sum(
                            v[j] * tf.tanh(hidden_features[j] + y), [2, 3])
                        a = tf.nn.softmax(s)
                        attn_weights.append(a)
                        # Now calculate the attention-weighted vector d.
                        d = tf.reduce_sum(
                            tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                            [1, 2])
                        ds.append(tf.reshape(d, [-1, attn_size]))
                return attn_weights, ds

            batch_attn_size = tf.stack([batch_size, attn_size])
            attns = [tf.zeros(batch_attn_size, dtype=dtype)
                     for _ in xrange(num_heads)]
            # 确保设置了注意向量的第二形状。
            for a in attns:
                a.set_shape([None, attn_size])

            # 循环通过encoder_output
            attention_encoder_outputs = list()
            sequence_attention_weights = list()
            for i in xrange(len(encoder_outputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                if i == 0:
                    with tf.variable_scope("Initial_Decoder_Attention"):
                        initial_state = linear(encoder_state, output_size, True)
                    attn_weights, ds = attention(initial_state)
                else:
                    attn_weights, ds = attention(encoder_outputs[i])
                output = tf.concat([ds[0], encoder_outputs[i]], 1)
                # NOTE: here we temporarily assume num_head = 1
                with tf.variable_scope("AttnRnnOutputProjection"):
                    logit = linear(output, num_decoder_symbols, True)
                attention_encoder_outputs.append(logit)
                # NOTE: here we temporarily assume num_head = 1
                sequence_attention_weights.append(attn_weights[0])
                # NOTE: here we temporarily assume num_head = 1
    else:
        print('使用无注意力RNN模型')
        with tf.variable_scope(scope or "non-attention_RNN"):
            attention_encoder_outputs = list()
            sequence_attention_weights = list()

            # copy over logits once out of sequence_length
            if encoder_outputs[0].get_shape().ndims != 1:
                (fixed_batch_size, output_size) = encoder_outputs[0].get_shape().with_rank(2)
            else:
                fixed_batch_size = encoder_outputs[0].get_shape().with_rank_at_least(1)[0]

            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value
            else:
                batch_size = tf.shape(encoder_outputs[0])[0]
            if sequence_length is not None:
                sequence_length = tf.to_int32(sequence_length)
            if sequence_length is not None:  # Prepare variables
                zero_logit = tf.zeros(
                    tf.stack([batch_size, num_decoder_symbols]), encoder_outputs[0].dtype)
                zero_logit.set_shape(
                    tensor_shape.TensorShape([fixed_batch_size.value,
                                              num_decoder_symbols]))
                min_sequence_length = tf.reduce_min(sequence_length)
                max_sequence_length = tf.reduce_max(sequence_length)

            # reuse = False
            for time, input_ in enumerate(encoder_outputs):
                if time > 0:
                    tf.get_variable_scope().reuse_variables()
                    # reuse = True
                # pylint: disable=cell-var-from-loop
                # call_cell = lambda: cell(input_, state)
                generate_logit = lambda: linear(encoder_outputs[time],
                                                num_decoder_symbols,
                                                True)
                # pylint: enable=cell-var-from-loop
                if sequence_length is not None:
                    logit = _step(time, sequence_length, min_sequence_length,
                                  max_sequence_length, zero_logit, generate_logit)
                else:
                    logit = generate_logit
                attention_encoder_outputs.append(logit)

    return attention_encoder_outputs, sequence_attention_weights


def generate_sequence_output(num_encoder_symbols,
                             encoder_outputs,
                             encoder_state,
                             targets,
                             sequence_length,
                             num_decoder_symbols,
                             weights,
                             buckets,
                             softmax_loss_function=None,
                             per_example_loss=False,
                             name=None,
                             use_attention=False):
    if len(targets) < buckets[-1][1]:
        raise ValueError("目标的长度(%d)必须至少是最后一个桶的长度(%d)。"
                         % (len(targets), buckets[-1][1]))

    all_inputs = encoder_outputs + targets + weights
    with tf.name_scope(name, "model_with_buckets", all_inputs):
        with tf.variable_scope("decoder_sequence_output", reuse=None):
            logits, attention_weights = attention_rnn(encoder_outputs,
                                                      encoder_state,
                                                      num_decoder_symbols,
                                                      sequence_length,
                                                      use_attention=use_attention)
            if per_example_loss is None:
                assert len(logits) == len(targets)
                # We need to make target and int64-tensor and set its shape.
                bucket_target = [tf.reshape(tf.to_int64(x), [-1]) for x in targets]
                crossent = sequence_loss_by_example(
                    logits, bucket_target, weights,
                    softmax_loss_function=softmax_loss_function)
            else:
                assert len(logits) == len(targets)
                bucket_target = [tf.reshape(tf.to_int64(x), [-1]) for x in targets]
                crossent = sequence_loss(
                    logits, bucket_target, weights,
                    softmax_loss_function=softmax_loss_function)

    return logits, crossent
