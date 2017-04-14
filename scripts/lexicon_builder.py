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
import os

FLAGS = gflags.FLAGS

gflags.DEFINE_string('filepath', 'data/glove.6B/glove.6B.50d.train.txt', 'File path to pretrained embedding.')
gflags.DEFINE_boolean('lower', False, 'True to lowercase the lexicon.')
gflags.DEFINE_integer('min_freq', 20, 'Minimum number of occurencies for a char.')

def main(_):
  with open(FLAGS.filepath, 'r') as f:
    vocab = list()
    for line in f:
      if not line: continue
      word = line.split(' ')[0]
      vocab.append(word if not FLAGS.lower else word.lower())
  pass

  dstdir = os.path.dirname(FLAGS.filepath)
  with open(os.path.join(dstdir, 'words.txt'), 'w') as f:
    f.write("%d\n" % len(vocab))
    f.write("\n".join(vocab))

  chars = dict()
  for word in vocab:
    for char in word:
      if char in chars:
        chars[char] += 1
      else:
        chars[char] = 1


  chars = {k: v for k, v in chars.items() if v >= FLAGS.min_freq}

  with open(os.path.join(dstdir, 'chars.txt'), 'w') as f:
    f.write("%d\n" % len(chars))
    f.write("\n".join(sorted(chars.keys())))

if __name__ == '__main__':
  app.run()
