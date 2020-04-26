# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:23:37 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils
import multi_task_model

import subprocess
import stat


# 配置参数
# tf.app.flags.DEFINE_float("learning_rate", 0.1, "学习率.")
# tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
#                          "学习率下降了这么多。")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "剪切梯度标准。")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "每一个模型层的大小。")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "词向量大小。")
tf.app.flags.DEFINE_integer("num_layers", 1, "模型层数。")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "最大词汇数。")
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "最大标签词汇数。")
tf.app.flags.DEFINE_string("data_dir", r"./data/ATIS_samples", "数据文件。")
tf.app.flags.DEFINE_string("train_dir", r"model_tmp", "训练文件")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "限制训练数据的大小 (0: 无限制)")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "每个检查点需要多少个训练步骤。")
tf.app.flags.DEFINE_integer("max_training_steps", 2000,
                            "最大训练步骤。")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "最大测试数据量。")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "使用基于RNN的注意力机制。")
tf.app.flags.DEFINE_integer("max_sequence_length", 50,
                            "最大序列长度。")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "退出保持单元输入和输出率。")
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True,
                            "使用双向RNN。")
tf.app.flags.DEFINE_string("task", 'joint', "可选项: intent; tagging; joint")
FLAGS = tf.app.flags.FLAGS

# 指定会话长度
if FLAGS.max_sequence_length == 0:
    print('请指出最大序列长度。 退出')
    exit()

# 指定任务
if FLAGS.task is None:
    print('请指出任务可运行的可选项：intent; tagging; joint')
    exit()

task = dict({'intent': 0, 'tagging': 0, 'joint': 0})
if FLAGS.task == 'intent':
    task['intent'] = 1
elif FLAGS.task == 'tagging':
    task['tagging'] = 1
elif FLAGS.task == 'joint':
    task['intent'] = 1
    task['tagging'] = 1
    task['joint'] = 1

# _buckets = [(3, 10), (10, 25)]
_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]


# 度量函数使用conlleval.pl
def conlleval(p, g, w, filename):
    """
    INPUT:
    p :: 预测
    g :: 真实状况
    w :: 对应词

    OUTPUT:
    filename :: 写入预测的文件的名称。 它将作为conlleval.pl脚本的输入，用于根据准确率/召回率和f1分数计算性能
    """
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')

    # remove the ending \n on last line
    f.writelines(out[:-1])
    f.close()

    return get_perf(out, filename)


def get_perf(out, filename):
    """ 运行conllev.pl perl脚本获得准确率/召回率和F1得分 """
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.pl'
    # 授予执行权限
    os.chmod(_conlleval, stat.S_IRWXU)

    proc = subprocess.Popen(["perl",
                            _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            universal_newlines=True)

    # 子程序返回标准输出与标准错误
    stdout, _ = proc.communicate(''.join(open(filename).readlines()))
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    # 返回准确率，召回率，f1得分
    return {'p': precision, 'r': recall, 'f1': f1score}


def read_data(source_path, target_path, label_path, max_size=None):
    """
    从源文件和目标文件中读取数据并放入存储段。

    Args:
    source_path: 带有单词序列标记id的文件的路径。
    target_path: 带有标签序列标记id的文件的路径。
        它必须与源文件对齐: 第n行包含source_path的第n行所需的输出。
    label_path: 使用意图标签的标记id的文件的路径。
    max_size: 最大读取行数，其他所有行将被忽略;
        如果是0或者None，数据文件将全部读取(无限制).

    Returns:
    data_set: len(_buckets)的长度; data_set[n] 包括一个
        (source, target, label)元组列表，为从第n个桶中读取提供的数据文件,
         例如,len(source) < _buckets[n][0]和
         len(target) < _buckets[n][1];源, 目标, 带有标签序列标记id
    """


    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            with tf.gfile.GFile(label_path, mode="r") as label_file:
                source = source_file.readline()
                target = target_file.readline()
                label = label_file.readline()
                counter = 0
                while source and target and label and (not max_size or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                        print("正在读取数据行 %d" % counter)
                        sys.stdout.flush()
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    label_ids = [int(x) for x in label.split()]
                    # target_ids.append(data_utils.EOS_ID)
                    for bucket_id, (source_size, target_size) in enumerate(_buckets):
                        if len(source_ids) < source_size and len(target_ids) < target_size:
                            data_set[bucket_id].append([source_ids, target_ids, label_ids])
                            break
                    source = source_file.readline()
                    target = target_file.readline()
                    label = label_file.readline()

    # 每个单元的三个输出：source_ids, target_ids, label_ids
    return data_set


def create_model(session,
                 source_vocab_size, 
                 target_vocab_size, 
                 label_vocab_size):
    """创建模型并在会话中初始化或者加载参数。"""
    with tf.variable_scope("model", reuse=None):
        model_train = multi_task_model.MultiTaskModel(
            source_vocab_size,
            target_vocab_size,
            label_vocab_size,
            _buckets,
            FLAGS.word_embedding_size,
            FLAGS.size, FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            use_lstm=True,
            forward_only=False,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn,
            task=task)
    with tf.variable_scope("model", reuse=True):
        model_test = multi_task_model.MultiTaskModel(
            source_vocab_size,
            target_vocab_size,
            label_vocab_size,
            _buckets,
            FLAGS.word_embedding_size,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            use_lstm=True,
            forward_only=True,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn,
            task=task)

    # 判断是否添加模型
    checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if checkpoint:
        print("读取模型参数： %s" % checkpoint.model_checkpoint_path)
        model_train.saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        print("使用新的参数创建模型。")
        session.run(tf.global_variables_initializer())
    return model_train, model_test


def train():
    print('应用参数:')
    for k, v in FLAGS.__dict__['__flags'].items():
        print('%s: %s' % (k, str(v)))

    print("准备数据 %s" % FLAGS.data_dir)
    vocab_path = ''
    tag_vocab_path = ''
    label_vocab_path = ''
    date_set = data_utils.prepare_multi_task_data(
        FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)
    in_seq_train, out_seq_train, label_train = date_set[0]
    in_seq_dev, out_seq_dev, label_dev = date_set[1]
    in_seq_test, out_seq_test, label_test = date_set[2]
    vocab_path, tag_vocab_path, label_vocab_path = date_set[3]

    result_dir = FLAGS.train_dir + '/test_results'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    current_tagging_valid_out_file = result_dir + '/tagging.valid.hyp.txt'
    current_tagging_test_out_file = result_dir + '/tagging.test.hyp.txt'

    vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
    tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)
    label_vocab, rev_label_vocab = data_utils.initialize_vocab(label_vocab_path)

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.23),
        # device_count = {'gpu': 2}
    )

    with tf.Session(config=config) as sess:
        # 创建模型。
        print("最大序列长度: %d." % _buckets[0][0])
        print("创建%d单元的%d层。" % (FLAGS.num_layers, FLAGS.size))

        model, model_test = create_model(sess,
                                         len(vocab),
                                         len(tag_vocab),
                                         len(label_vocab))
        print("创建模型 " +
              "source_vocab_size=%d, target_vocab_size=%d, label_vocab_size=%d."
              % (len(vocab), len(tag_vocab), len(label_vocab)))

        # 将数据读入桶中并计算桶的大小。
        print("读取 train/valid/test 数据 (训练集范围: %d)."
              % FLAGS.max_train_data_size)
        dev_set = read_data(in_seq_dev, out_seq_dev, label_dev)
        test_set = read_data(in_seq_test, out_seq_test, label_test)
        train_set = read_data(in_seq_train, out_seq_train, label_train)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # 这是一个训练循环。
        step_time, loss = 0.0, 0.0
        current_step = 0

        best_valid_score = 0
        best_test_score = 0
        while model.global_step.eval() < FLAGS.max_training_steps:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # 获取一个分支并执行一步
            start_time = time.time()
            batch_data = model.get_batch(train_set, bucket_id)
            encoder_inputs, tags, tag_weights, batch_sequence_length, labels = batch_data
            if task['joint'] == 1:
                step_outputs = model.joint_step(sess,
                                                encoder_inputs,
                                                tags,
                                                tag_weights,
                                                labels,
                                                batch_sequence_length,
                                                bucket_id,
                                                False)
                _, step_loss, tagging_logits, class_logits = step_outputs
            elif task['tagging'] == 1:
                step_outputs = model.tagging_step(sess,
                                                  encoder_inputs,
                                                  tags,
                                                  tag_weights,
                                                  batch_sequence_length,
                                                  bucket_id,
                                                  False)
                _, step_loss, tagging_logits = step_outputs
            elif task['intent'] == 1:
                step_outputs = model.classification_step(sess,
                                                         encoder_inputs,
                                                         labels,
                                                         batch_sequence_length,
                                                         bucket_id,
                                                         False)
                _, step_loss, class_logits = step_outputs

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # 有时，我们保存检查点、打印统计数据并运行evals。
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("全局步数 %d 每步时间 %.2fs 训练复杂度 %.2f"
                      % (model.global_step.eval(), step_time, perplexity))
                sys.stdout.flush()
                # 保存检查点和零计时器和损失。
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                # 模式: Eval, Test
                def run_valid_test(data_set, mode):
                    # 在开发/测试集上运行evals并打印准确性。
                    word_list = list()
                    ref_tag_list = list()
                    hyp_tag_list = list()
                    ref_label_list = list()
                    hyp_label_list = list()
                    correct_count = 0
                    accuracy = 0.0
                    tagging_eval_result = dict()
                    eval_loss = 0.0
                    count = 0
                    for bucket_id in xrange(len(_buckets)):
                        for i in xrange(len(data_set[bucket_id])):
                            count += 1
                            sample = model_test.get_one(data_set, bucket_id, i)
                            encoder_inputs, tags, tag_weights, sequence_length, labels = sample
                            tagging_logits = []
                            class_logits = []

                            step_loss = None
                            if task['joint'] == 1:
                                step_outputs = model_test.joint_step(sess,
                                                                     encoder_inputs,
                                                                     tags,
                                                                     tag_weights,
                                                                     labels,
                                                                     sequence_length,
                                                                     bucket_id,
                                                                     True)
                                _, step_loss, tagging_logits, class_logits = step_outputs
                            elif task['tagging'] == 1:
                                step_outputs = model_test.tagging_step(sess,
                                                                       encoder_inputs,
                                                                       tags,
                                                                       tag_weights,
                                                                       sequence_length,
                                                                       bucket_id,
                                                                       True)
                                _, step_loss, tagging_logits = step_outputs
                            elif task['intent'] == 1:
                                step_outputs = model_test.classification_step(sess,
                                                                              encoder_inputs,
                                                                              labels,
                                                                              sequence_length,
                                                                              bucket_id,
                                                                              True)
                                _, step_loss, class_logits = step_outputs
                            eval_loss += step_loss / len(data_set[bucket_id])
                            hyp_label = None
                            if task['intent'] == 1:
                                ref_label_list.append(rev_label_vocab[labels[0][0]])
                                hyp_label = np.argmax(class_logits[0], 0)
                                hyp_label_list.append(rev_label_vocab[hyp_label])
                                if labels[0] == hyp_label:
                                    correct_count += 1
                            if task['tagging'] == 1:
                                word_list.append([rev_vocab[x[0]] for x in
                                                  encoder_inputs[:sequence_length[0]]])
                                ref_tag_list.append([rev_tag_vocab[x[0]] for x in
                                                     tags[:sequence_length[0]]])
                                hyp_tag_list.append(
                                    [rev_tag_vocab[np.argmax(x)] for x in
                                     tagging_logits[:sequence_length[0]]])

                    accuracy = float(correct_count) * 100 / count
                    if task['intent'] == 1:
                        print("  %s 准确性: %.2f%% %d/%d"
                              % (mode, accuracy, correct_count, count))
                        sys.stdout.flush()
                    if task['tagging'] == 1:
                        taging_out_file = None
                        if mode == 'Eval':
                            taging_out_file = current_tagging_valid_out_file
                        elif mode == 'Test':
                            taging_out_file = current_tagging_test_out_file
                        tagging_eval_result = conlleval(hyp_tag_list,
                                                        ref_tag_list,
                                                        word_list,
                                                        taging_out_file)
                        print("  %s f1-score: %.2f%%" % (mode, tagging_eval_result['f1']))
                        sys.stdout.flush()
                    return accuracy, tagging_eval_result

                # valid
                valid_accuracy, valid_tagging_result = run_valid_test(dev_set, 'Eval')
                if task['tagging'] == 1 \
                        and valid_tagging_result['f1'] > best_valid_score:
                    best_valid_score = valid_tagging_result['f1']
                    # 保存最好的输出文件
                    subprocess.call(['mv',
                                     current_tagging_valid_out_file,
                                     current_tagging_valid_out_file + '.best_f1_%.2f'
                                     % best_valid_score])
                # 测试，在每个验证后运行测试，以供开发之用。
                test_accuracy, test_tagging_result = run_valid_test(test_set, 'Test')
                if task['tagging'] == 1 \
                        and test_tagging_result['f1'] > best_test_score:
                    best_test_score = test_tagging_result['f1']
                    # 保存最好的输出文件
                    subprocess.call(['mv',
                                     current_tagging_test_out_file,
                                     current_tagging_test_out_file + '.best_f1_%.2f'
                                     % best_test_score])


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
