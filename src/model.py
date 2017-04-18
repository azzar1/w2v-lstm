#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Andrea Azzarone <azzaronea@gmail.com>
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import numpy as np
import os
from progress.bar import Bar
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_fold.public.blocks as td

flags = tf.app.flags
FLAGS = flags.FLAGS
td.define_plan_flags(default_plan_name='w2v-lstm')

# Gnerig options
flags.DEFINE_string('train_set', 'data/glove.6B/glove.6B.50d.train.txt', 'Filepath to trainset.')
flags.DEFINE_string('dev_set', 'data/glove.6B/glove.6B.50d.dev.txt', 'Filepath to devset.')
flags.DEFINE_string('test_set', '', 'Filepath to testset.')
flags.DEFINE_string('test_output', '', 'Filepath to output file.')

flags.DEFINE_integer('char_embedding_size', 8, 'Size of the char embeddings.')
flags.DEFINE_integer('word_embedding_size', 50, 'Size of the word embeddings.')

# LSTM specific flags
flags.DEFINE_integer('num_units', 150, 'Size of LSTM memory.')
flags.DEFINE_float('forget_bias', 1.0, 'The bias added to forget gates.')
flags.DEFINE_boolean('layer_norm', True, 'If True, layer normalization will be applied.')
flags.DEFINE_float('norm_gain', 1.0, 'The layer normalization gain initial value. If layer_norm has been set to False, this argument will be ignored.')
flags.DEFINE_float('norm_shift', 0.0, 'The layer normalization shift initial value. If layer_norm has been set to False, this argument will be ignored.')
flags.DEFINE_float('dropout_keep_prob', 1.0, 'Float between 0 and 1 representing the recurrent dropout probability value. If 1.0, no dropout will be applied.')
flags.DEFINE_integer('dropout_prob_seed', None, 'The randomness seed.')


def load_char2index():
    filepath = os.path.join(os.path.dirname(FLAGS.train_set), 'chars.txt')
    assert os.path.exists(filepath), "%s does not exists." % filepath

    with codecs.open(filepath, encoding='utf-8') as f:
        lines = f.read().splitlines()
        assert len(
            lines) > 0, "Invald char vocabulary file: %s has invalid lenght." % filepath

        char2index = dict()
        for i, char in enumerate(lines[1:]):
            char2index[char] = i
        return char2index

def char2index(vocab, char):
  if char in vocab:
      return vocab[char]
  else:
      return len(vocab)

def load_trainset():
    return load_set(FLAGS.train_set)


def load_devset():
    return load_set(FLAGS.dev_set)

def load_testset():
    filename = FLAGS.test_set

    words = []
    with codecs.open(filename, encoding='utf-8') as f:
        bar = Bar('Loading dataset %s' % filename, max=get_size_set(filename))
        for word in f:
            bar.next()
            word = word.strip()
            if not word: continue
            words.append(word)
    bar.finish()
    return words

def get_size_set(filename):
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return len(lines)


def load_set(filename):
    words = []
    embeddings = []

    with codecs.open(filename, encoding='utf-8') as f:
        bar = Bar('Loading dataset %s' % filename, max=get_size_set(filename))
        for line in f:
            bar.next()
            line = line.strip()
            if not line: continue
            word, vec = line.split(u' ', 1)
            emb = np.array(vec.split(), dtype=np.float32)
            if len(emb) != FLAGS.word_embedding_size: continue
            words.append(word)
            embeddings.append(np.array(vec.split(), dtype=np.float32))
    bar.finish()
    return words, embeddings

def setup_plan(plan):
    # Save used paramteres
    with open(os.path.join(FLAGS.logdir_base, str(FLAGS.run_id)+'-params.txt'), 'w') as f:
        f.write(str(FLAGS.__dict__['__flags']))

    # Convert a word in a list of integers using chars.txt
    vocab = load_char2index()
    word2integers = td.InputTransform(lambda s: [char2index(vocab, c) for c in s])

    # Create a placeholder for dropout, if we are in train mode.
    keep_prob = tf.placeholder_with_default(1.0, [], name='keep_prob')

    # The lstm cell
    fw_char_cell = td.ScopedLayer(
        tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=FLAGS.num_units,
                                              forget_bias=FLAGS.forget_bias,
                                              layer_norm=FLAGS.layer_norm,
                                              norm_gain=FLAGS.norm_gain,
                                              norm_shift=FLAGS.norm_shift,
                                              dropout_keep_prob=keep_prob,
                                              dropout_prob_seed=FLAGS.dropout_prob_seed), 'char_cell')

    bw_char_cell = td.ScopedLayer(
        tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=FLAGS.num_units,
                                              forget_bias=FLAGS.forget_bias,
                                              layer_norm=FLAGS.layer_norm,
                                              norm_gain=FLAGS.norm_gain,
                                              norm_shift=FLAGS.norm_shift,
                                              dropout_keep_prob=keep_prob,
                                              dropout_prob_seed=FLAGS.dropout_prob_seed), 'char_cell')

    # int -> char embedding (+1 for unk values)
    char_embedding = td.Scalar('int32') >> td.Function(td.Embedding(len(vocab) + 1, FLAGS.char_embedding_size))

    # word -> matrix of char embeddings
    word2matrix = word2integers >> td.Map(char_embedding)

    # word -> word embedding
    fw_pass = (td.RNN(fw_char_cell) >>
               td.GetItem(1) >> td.GetItem(1))

    reverse_word = td.Slice(step=-1)
    bw_pass = (reverse_word >>
               td.RNN(fw_char_cell) >>
               td.GetItem(1) >> td.GetItem(1))

    # Bidirectional lstm
    word_embedding = (word2matrix >>
                      td.AllOf(fw_pass, bw_pass) >>
                      td.Concat() >>
                      td.Function(td.FC(num_units_out=FLAGS.word_embedding_size, activation=tf.tanh)))
    
    sess = tf.InteractiveSession()

    if plan.mode == plan.mode_keys.INFER:
        # In inference mode, we run the model directly on words.
        plan.compiler = td.Compiler.create(word_embedding)
        embedding_pred, = plan.compiler.output_tensors
    else:
        # In training/eval mode, we run the model on (word, embedding) pairs.
        plan.compiler = td.Compiler.create(
            td.Record((word_embedding, td.Vector(FLAGS.word_embedding_size))))
        embedding_pred, embedding_true = plan.compiler.output_tensors

    if plan.mode == plan.mode_keys.INFER:
        #results = list()
        def key_fn(sample):
            return sample
        def results_fn(results):
            with codecs.open(FLAGS.test_output, 'w', encoding='utf-8-sig') as f:
                for result in results:
                    word, embedding = result
                    f.write(word)
                    f.write(u' ')
                    f.write(u' '.join(str(x).encode("utf-8").decode("utf-8") for x in embedding[0].tolist()))
                    f.write(u'\n')
        plan.examples = load_testset()
        plan.outputs = [embedding_pred]
        plan.key_fn = key_fn
        plan.results_fn = results_fn
    else:
        trainset_words, trainset_embeddings = load_trainset()
        devset_words, devset_embeddings = load_devset()

        # Create loss tensor, and add it to the plan.
        loss_x = tf.losses.mean_squared_error(embedding_true, embedding_pred)
        loss = tf.Print(loss_x, [learning_rate, loss_x])
        plan.losses['mse'] = loss

        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate, plan.global_step,
                                                   100, 0.96, staircase=True)

        optr = tf.train.GradientDescentOptimizer(learning_rate)
        plan.train_op = optr.minimize(plan.losses['cos'], plan.global_step)

        # collect all trainable variables
        #tvars = tf.trainable_variables()
        #grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.0)

        #optimizer = tf.train.AdagradOptimizer(0.09)
        #plan.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=plan.global_step)

        if plan.mode == plan.mode_keys.TRAIN:
            plan.examples = zip(trainset_words, trainset_embeddings)
            plan.dev_examples = zip(devset_words, devset_embeddings)
            # Turn dropout on for training, off for validation.
            plan.train_feeds[keep_prob] = FLAGS.dropout_keep_prob
            plan.report_loss = lambda step, loss: print(step, loss)
        else:
            assert plan.mode == plan.mode_keys.EVAL
            # We evaluate on devset because we don't have a true testset.
            plan.examples = zip(devset_words, devset_embeddings)


def main(_):
    assert 0 < FLAGS.dropout_keep_prob <= 1, '--keep_prob must be in (0, 1]'
    td.Plan.create_from_flags(setup_plan).run()


if __name__ == '__main__':
    tf.app.run()
