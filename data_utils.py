# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:33:32 2016

@author: Bing Liu (liubing@cmu.edu)

Prepare data for multi-task RNN model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from tensorflow.python.platform import gfile

# 特殊的词汇符号——我们总是把它们放在开头。
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

START_VOCAB_dict = dict()
START_VOCAB_dict['with_padding'] = [_PAD, _UNK]
START_VOCAB_dict['no_padding'] = [_UNK]


PAD_ID = 0

UNK_ID_dict = dict()
UNK_ID_dict['with_padding'] = 1
UNK_ID_dict['no_padding'] = 0

# 用于标记的正则表达式。
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
    """非常基本的记号赋予器:将句子分割成记号列表。"""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def naive_tokenizer(sentence):
    """Naive tokenizer: 将句子按空格分成一系列标记。"""
    return sentence.split()


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """从数据文件创建词汇表文件(如果它还不存在)。

      假设数据文件每行包含一个句子。每个句子都是标记化的，数字是规范化的(如果设置了normalize_digits)。
      词汇表包含max_vocabulary_size最频繁的令牌.
      我们以one-token-per-line的形式写入vocabulary_path，所以第一行令牌的id=0,第二行id=1,等等。

      Args:
        vocabulary_path: 将要创建词汇表的路径。
        data_path: 将用于创建词汇表的数据文件。
        max_vocabulary_size: 限制所创建词汇表的大小。
        tokenizer: 用于标记每个数据语句的函数;
            如果是None，则将使用basic_tokenizer。
        normalize_digits: Boolean; 如果是true, 所有的数字将被0s替换。
      """
    if not gfile.Exists(vocabulary_path):
        print("从数据 %s 创建词汇表 %s " % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="r") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("加工线 %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    # 把数字全部替换为0
                    # word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
                    word = w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

            # 频率由大到小排序
            vocab_list = START_VOCAB_dict['with_padding'] + sorted(
                vocab, key=vocab.get, reverse=True)

            # 词汇库数量超过最大词汇表大小
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")


def initialize_vocab(vocabulary_path):
    """从文件初始化词汇表。

      我们假设词汇表每行存储一个项目，所以一个文件:
        dog
        cat
      再次回表中的结果是 {"dog": 0, "cat": 1},
      而且这个函数也会返回相反的词汇 ["dog", "cat"]。

      Args:
        vocabulary_path: 包含词汇表的文件的路径。

      Returns:
        a pair: 词汇表 (将字符串映射到整数的一个词典), 和
        相反的词汇 (一个列表, 反转了词汇表映射).

      Raises:
        ValueError: 如果提供的词汇表_path不存在。
      """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("没有找到词汇表文件%s。", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, UNK_ID,
                          tokenizer=None, normalize_digits=True):
    """将字符串转换为表示令牌id的整数列表。

      例如, 句子 "I have a dog" 会被标注为["I", "have", "a", "dog"]
      和词典 {"I": 1, "have": 2, "a": 4, "dog": 7"}
       这个函数会返回 [1, 2, 4, 7].

      Args:
        sentence: 一个字符串，要转换为令牌id的句子。
        vocabulary: 字典将记号映射到整数。
        tokenizer: 用来标记每个句子的函数;
            如果是None, 将会使用basic_tokenizer.
        normalize_digits: Boolean; 如果是true,所有的数字会被替换为0s.

      Returns:
        一个整数列表，句子的标记id。
      """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # 在查找词汇表中的单词之前，将数字归一化为0。
    # return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True, use_padding=True):
    """对数据文件进行标记，并使用给定的词汇表文件转换为标记id。

      这个函数从data_path中一行一行地加载数据, 调用上面的sentence_to_token_ids,
       然后把结果保存到target_path. 有关标记id格式的详细信息，请参阅对sentence_to_token_ids的注释。

      Args:
        data_path: 一行一句格式数据的文件路径。
        target_path: 带有令牌id的文件将在其中创建的路径。
        vocabulary_path: 词汇表文件的路径。
        tokenizer: 用来标记每个句子的函数;
            如果时None, 会使用basic_tokenizer。
        normalize_digits: Boolean; 如果是true, 所以的数字会替换为0s.
        use_padding:
      """
    if not gfile.Exists(target_path):
        print("标记数据 %s" % data_path)
        vocab, _ = initialize_vocab(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    if use_padding:
                        UNK_ID = UNK_ID_dict['with_padding']
                    else:
                        UNK_ID = UNK_ID_dict['no_padding']
                    token_ids = sentence_to_token_ids(line, vocab, UNK_ID, tokenizer,
                                                      normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def create_label_vocab(vocabulary_path, data_path):
    if not gfile.Exists(vocabulary_path):
        print("从数据%s创建词汇表%s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="r") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("加工线 %d" % counter)
                label = line.strip()
                vocab[label] = 1
            label_list = START_VOCAB_dict['no_padding'] + sorted(vocab)
            with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for k in label_list:
                    vocab_file.write(k + "\n")


def prepare_multi_task_data(data_dir, in_vocab_size, out_vocab_size):
    train_path = data_dir + '/train/train'
    dev_path = data_dir + '/valid/valid'
    test_path = data_dir + '/test/test'
    
    # 创建适当大小的词汇表。
    in_vocab_path = os.path.join(data_dir, "in_vocab_%d.txt" % in_vocab_size)
    out_vocab_path = os.path.join(data_dir, "out_vocab_%d.txt" % out_vocab_size)
    label_path = os.path.join(data_dir, "label.txt")
    
    create_vocabulary(in_vocab_path, 
                      train_path + ".seq.in", 
                      in_vocab_size, 
                      tokenizer=naive_tokenizer)
    create_vocabulary(out_vocab_path, 
                      train_path + ".seq.out", 
                      out_vocab_size, 
                      tokenizer=naive_tokenizer)
    create_label_vocab(label_path, train_path + ".label")
    
    # 为训练数据创建令牌id。
    in_seq_train_ids_path = train_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_train_ids_path = train_path + (".ids%d.seq.out" % out_vocab_size)
    label_train_ids_path = train_path + ".ids.label"

    data_to_token_ids(train_path + ".seq.in", 
                      in_seq_train_ids_path, 
                      in_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(train_path + ".seq.out", 
                      out_seq_train_ids_path, 
                      out_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(train_path + ".label", 
                      label_train_ids_path, 
                      label_path, 
                      normalize_digits=False, 
                      use_padding=False)
    
    # 为开发数据创建令牌id。
    in_seq_dev_ids_path = dev_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_dev_ids_path = dev_path + (".ids%d.seq.out" % out_vocab_size)
    label_dev_ids_path = dev_path + ".ids.label"

    data_to_token_ids(dev_path + ".seq.in", 
                      in_seq_dev_ids_path, 
                      in_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(dev_path + ".seq.out", 
                      out_seq_dev_ids_path, 
                      out_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(dev_path + ".label", 
                      label_dev_ids_path, 
                      label_path, 
                      normalize_digits=False, 
                      use_padding=False)
    
    # 为测试数据创建令牌id。
    in_seq_test_ids_path = test_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_test_ids_path = test_path + (".ids%d.seq.out" % out_vocab_size)
    label_test_ids_path = test_path + ".ids.label"
    
    data_to_token_ids(test_path + ".seq.in", 
                      in_seq_test_ids_path, 
                      in_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(test_path + ".seq.out", 
                      out_seq_test_ids_path, 
                      out_vocab_path, 
                      tokenizer=naive_tokenizer)
    data_to_token_ids(test_path + ".label", 
                      label_test_ids_path, 
                      label_path, 
                      normalize_digits=False, 
                      use_padding=False)
    
    return [(in_seq_train_ids_path, out_seq_train_ids_path, label_train_ids_path),
            (in_seq_dev_ids_path, out_seq_dev_ids_path, label_dev_ids_path),
            (in_seq_test_ids_path, out_seq_test_ids_path, label_test_ids_path),
            (in_vocab_path, out_vocab_path, label_path)]
