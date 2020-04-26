# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 17:28:22 2016

@author: Bing Liu (liubing@cmu.edu)

使用注意力机制的多任务RNN模型。
  - 在Tensorflow seq2seq_model.py示例的基础上开发的:
    https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py
  - 请注意，此示例代码不包括输出标签依赖项建模。
    可以在tensorflow seq2seq.py示例中的rnn_decoder函数中添加一个循环函数，
    将嵌入的发出标签反馈给RNN状态。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


import data_utils
import seq_labeling
import seq_classification
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import static_rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn


class MultiTaskModel(object):
    def __init__(self,
                 source_vocab_size,
                 tag_vocab_size,
                 label_vocab_size,
                 buckets,
                 word_embedding_size,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 dropout_keep_prob=1.0,
                 use_lstm=False,
                 bidirectional_rnn=True,
                 num_samples=1024,
                 use_attention=False,
                 task=None,
                 forward_only=False):
        self.source_vocab_size = source_vocab_size
        self.tag_vocab_size = tag_vocab_size
        self.label_vocab_size = label_vocab_size
        self.word_embedding_size = word_embedding_size
        self.cell_size = size
        self.num_layers = num_layers
        self.buckets = buckets
        self.batch_size = batch_size
        self.bidirectional_rnn = bidirectional_rnn
        self.global_step = tf.Variable(0, trainable=False)

        # 如果我们使用采样softmax，我们需要一个输出投影。
        softmax_loss_function = None

        # 为我们的RNN创建内部的多层单元。
        def create_cell():
            if not forward_only and dropout_keep_prob < 1.0:
                single_cell = lambda: BasicLSTMCell(self.cell_size)
                cell = MultiRNNCell([single_cell() for _ in range(self.num_layers)])
                cell = DropoutWrapper(cell,
                                      input_keep_prob=dropout_keep_prob,
                                      output_keep_prob=dropout_keep_prob)
            else:
                single_cell = lambda: BasicLSTMCell(self.cell_size)
                cell = MultiRNNCell([single_cell() for _ in range(self.num_layers)])
            return cell

        self.cell_fw = create_cell()
        self.cell_bw = create_cell()

        # 输入源。
        self.encoder_inputs = []
        self.tags = []
        self.tag_weights = []
        self.labels = []
        self.sequence_length = tf.placeholder(tf.int32, [None],
                                              name="sequence_length")

        for i in xrange(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1]):
            self.tags.append(tf.placeholder(tf.float32, shape=[None],
                                            name="tag{0}".format(i)))
            self.tag_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                   name="weight{0}".format(i)))
        self.labels.append(tf.placeholder(tf.float32, shape=[None], name="label"))

        base_rnn_output = self.generate_rnn_output()
        encoder_outputs, encoder_state, attention_states = base_rnn_output

        if task['tagging'] == 1:
            seq_labeling_outputs = seq_labeling.generate_sequence_output(
                self.source_vocab_size,
                encoder_outputs,
                encoder_state,
                self.tags,
                self.sequence_length,
                self.tag_vocab_size,
                self.tag_weights,
                buckets,
                softmax_loss_function=softmax_loss_function,
                use_attention=use_attention)
            self.tagging_output, self.tagging_loss = seq_labeling_outputs
        if task['intent'] == 1:
            seq_intent_outputs = seq_classification.generate_single_output(
                encoder_state,
                attention_states,
                self.sequence_length,
                self.labels,
                self.label_vocab_size,
                buckets,
                softmax_loss_function=softmax_loss_function,
                use_attention=use_attention)
            self.classification_output, self.classification_loss = seq_intent_outputs

        if task['tagging'] == 1:
            self.loss = self.tagging_loss
        elif task['intent'] == 1:
            self.loss = self.classification_loss

        # 梯度和SGD更新操作对模型进行训练。
        params = tf.trainable_variables()
        if not forward_only:
            opt = tf.train.AdamOptimizer()
            gradients = None
            if task['joint'] == 1:
                # 反向传播意图和标记损失，可以进一步调整这两种成本的权重。
                gradients = tf.gradients([self.tagging_loss, self.classification_loss],
                                         params)
            elif task['tagging'] == 1:
                gradients = tf.gradients(self.tagging_loss, params)
            elif task['intent'] == 1:
                gradients = tf.gradients(self.classification_loss, params)

            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norm = norm
            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def generate_rnn_output(self):
        """
        生成以词嵌入为输入的RNN状态输出
        """
        with tf.variable_scope("generate_seq_output"):
            if self.bidirectional_rnn:
                embedding = tf.get_variable("embedding",
                                            [self.source_vocab_size,
                                             self.word_embedding_size])
                encoder_emb_inputs = list()
                encoder_emb_inputs = [tf.nn.embedding_lookup(embedding, encoder_input) \
                                      for encoder_input in self.encoder_inputs]
                rnn_outputs = static_bidirectional_rnn(self.cell_fw,
                                                       self.cell_bw,
                                                       encoder_emb_inputs,
                                                       sequence_length=self.sequence_length,
                                                       dtype=tf.float32)
                encoder_outputs, encoder_state_fw, encoder_state_bw = rnn_outputs
                # with state_is_tuple = True, if num_layers > 1,
                # 这里我们简单地使用最后一层的状态作为编码器的状态
                state_fw = encoder_state_fw[-1]
                state_bw = encoder_state_bw[-1]
                encoder_state = tf.concat([tf.concat(state_fw, 1),
                                           tf.concat(state_bw, 1)], 1)
                top_states = [tf.reshape(e, [-1, 1, self.cell_fw.output_size \
                                             + self.cell_bw.output_size])
                              for e in encoder_outputs]
                attention_states = tf.concat(top_states, 1)
            else:
                embedding = tf.get_variable("embedding",
                                            [self.source_vocab_size,
                                             self.word_embedding_size])
                encoder_emb_inputs = list()
                encoder_emb_inputs = [tf.nn.embedding_lookup(embedding, encoder_input) \
                                      for encoder_input in self.encoder_inputs]
                rnn_outputs = static_rnn(self.cell_fw,
                                         encoder_emb_inputs,
                                         sequence_length=self.sequence_length,
                                         dtype=tf.float32)
                encoder_outputs, encoder_state = rnn_outputs
                # with state_is_tuple = True, if num_layers > 1,
                # here we use the state from last layer as the encoder state
                state = encoder_state[-1]
                encoder_state = tf.concat(state, 1)
                top_states = [tf.reshape(e, [-1, 1, self.cell_fw.output_size])
                              for e in encoder_outputs]
                attention_states = tf.concat(top_states, 1)
            return encoder_outputs, encoder_state, attention_states

    def joint_step(self, session, encoder_inputs, tags, tag_weights,
                   labels, batch_sequence_length,
                   bucket_id, forward_only):
        """运行给定输入的联合模型的一个步骤。

        Args:
          session: 使用的tensorflow会话
          encoder_inputs: 要作为编码器输入的numpy int向量的列表。
          tags: 要作为解码器输入的numpy int向量的列表。
          tag_weights: 要作为标记权重提供的numpy浮动向量列表。
          labels: 要作为序列类标签提供的numpy int向量列表。
          bucket_id: 要使用模型的哪个桶。
          batch_sequence_length: 分支序列长度
          forward_only: 是后退一步，还是只向前一步。

        Returns:
          由梯度范数组成的三元组 (or None if we did not do backward),
          average perplexity, output tags, and output class label.

        Raises:
          ValueError: 如果encoder_input、decoder_input或target_weights的长度
          与指定的bucket_id的桶大小不一致。
        """
        # 检查大小是否匹配。
        encoder_size, tag_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(tags) != tag_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(tags), tag_size))
        if len(labels) != 1:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(labels), 1))

        input_feed = {}
        input_feed[self.sequence_length.name] = batch_sequence_length
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.tags[l].name] = tags[l]
            input_feed[self.tag_weights[l].name] = tag_weights[l]
        input_feed[self.labels[0].name] = labels[0]

        # Output feed: 这取决于我们是否倒退一步。
        if not forward_only:
            output_feed = [self.update,  # Update Op that does SGD.
                           self.gradient_norm,  # Gradient norm.
                           self.loss]  # Loss for this batch.
            for i in range(tag_size):
                output_feed.append(self.tagging_output[i])
            output_feed.append(self.classification_output[0])
        else:
            output_feed = [self.loss]
            for i in range(tag_size):
                output_feed.append(self.tagging_output[i])
            output_feed.append(self.classification_output[0])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[3:3 + tag_size], outputs[-1]
        else:
            return None, outputs[0], outputs[1:1 + tag_size], outputs[-1]

    def tagging_step(self, session, encoder_inputs, tags, tag_weights,
                     batch_sequence_length, bucket_id, forward_only):
        """运行标记模型的一个步骤，提供给定的输入。

        Args:
          session: 使用的tensorflow会话
          encoder_inputs: 要作为编码器输入的numpy int向量的列表。
          tags: 要作为解码器输入的numpy int向量的列表。
          tag_weights: 要作为标记权重提供的numpy浮动向量列表。
          batch_sequence_length: 分支序列长度
          bucket_id: 要使用模型的哪个桶。
          forward_only: 是后退一步，还是只向前一步。

        Returns:
          由梯度范数组成的二元组 (or None if we did not do backward),
          average perplexity, and the output tags.

        Raises:
          ValueError: 如果encoder_input、decoder_input或target_weights的长度
          与指定的bucket_id的桶大小不一致。
        """
        # 检查大小是否匹配。
        encoder_size, tag_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("编码器的长度必须等于篮中的长度,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(tags) != tag_size:
            raise ValueError("解码器的长度必须等于篮中的长度,"
                             " %d != %d." % (len(tags), tag_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.sequence_length.name] = batch_sequence_length
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.tags[l].name] = tags[l]
            input_feed[self.tag_weights[l].name] = tag_weights[l]

        # Output feed: 这取决于我们是否倒退一步。
        if not forward_only:
            output_feed = [self.update,  # Update Op that does SGD.
                           self.gradient_norm,  # Gradient norm.
                           self.loss]  # Loss for this batch.
            for i in range(tag_size):
                output_feed.append(self.tagging_output[i])
        else:
            output_feed = [self.loss]
            for i in range(tag_size):
                output_feed.append(self.tagging_output[i])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[3:3 + tag_size]
        else:
            return None, outputs[0], outputs[1:1 + tag_size]

    def classification_step(self, session, encoder_inputs, labels,
                            batch_sequence_length, bucket_id, forward_only):
        """运行intent分类模型的一个步骤，输入给定的输入。

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          labels: list of numpy int vectors to feed as sequence class labels.
          batch_sequence_length: batch_sequence_length
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the output class label.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, target_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.sequence_length.name] = batch_sequence_length
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.labels[0].name] = labels[0]

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.update,  # Update Op that does SGD.
                           self.gradient_norm,  # Gradient norm.
                           self.loss,  # Loss for this batch.
                           self.classification_output[0]]
        else:
            output_feed = [self.loss,
                           self.classification_output[0], ]

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[3]  # Gradient norm, loss, outputs.
        else:
            return None, outputs[0], outputs[1]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id):
        """从指定的桶中随机获取一批数据，为下一步做准备。

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs, labels = [], [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        batch_sequence_length_list = list()
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input, label = random.choice(data[bucket_id])
            batch_sequence_length_list.append(len(encoder_input))

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            # encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            encoder_inputs.append(list(encoder_input + encoder_pad))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input)
            decoder_inputs.append(decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)
            labels.append(label)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs = []
        batch_decoder_inputs = []
        batch_weights = []
        batch_labels = []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                #        if length_idx < decoder_size - 1:
                #          target = decoder_inputs[batch_idx][length_idx + 1]
                #        print (length_idx)
                if decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        batch_labels.append(
            np.array([labels[batch_idx][0]
                      for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
        return (batch_encoder_inputs, batch_decoder_inputs, batch_weights,
                batch_sequence_length, batch_labels)

    def get_one(self, data, bucket_id, sample_id):
        """从指定的桶中获取单个样本数据，准备执行步骤。

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs, labels = [], [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        batch_sequence_length_list = list()
        # for _ in xrange(self.batch_size):
        encoder_input, decoder_input, label = data[bucket_id][sample_id]
        batch_sequence_length_list.append(len(encoder_input))

        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        # encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
        encoder_inputs.append(list(encoder_input + encoder_pad))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input)
        decoder_inputs.append(decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)
        labels.append(label)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs = []
        batch_decoder_inputs = []
        batch_weights = []
        batch_labels = []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(1)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(1)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(1, dtype=np.float32)
            for batch_idx in xrange(1):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                #        if length_idx < decoder_size - 1:
                #          target = decoder_inputs[batch_idx][length_idx + 1]
                #        print (length_idx)
                if decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        batch_labels.append(
            np.array([labels[batch_idx][0]
                      for batch_idx in xrange(1)], dtype=np.int32))

        batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
        return (batch_encoder_inputs, batch_decoder_inputs, batch_weights,
                batch_sequence_length, batch_labels)
