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

from google.apputils import app
import gflags

import random
import os

FLAGS = gflags.FLAGS

gflags.DEFINE_string('filepath', 'data/glove.6B/glove.6B.50d.txt', 'File path to pretrained embedding.')
gflags.DEFINE_integer('ratio', 2, 'Ratio of words to be used as dev set.')

def main(_):
  with open(FLAGS.filepath, 'r') as f:
    lines = f.readlines()

    random.seed(0)
    random.shuffle(lines)

    num_words = len(lines)
    num_words_dev = int(num_words * FLAGS.ratio / 100)
    num_words_train = num_words - num_words_dev

    filepath, ext = os.path.splitext(FLAGS.filepath)
    trainset_path = filepath + '.train' + ext
    devset_path = filepath + '.dev' + ext

    with open(trainset_path, 'w') as tf:
      tf.write(''.join(lines[:num_words_train]))

    with open(devset_path, 'w') as df:
      df.write(''.join(lines[-num_words_dev:]))

if __name__ == '__main__':
  app.run()
