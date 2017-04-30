import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
from six.moves import cPickle
import random
import tensorflow as tf
import time

from model import Model


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('--word', type=str, default=u'apfelbrothauskirche',
                        help='Word for splitting')
    parser.add_argument('--device', type=str, default='/gpu:0',
                        help='device')
    args = parser.parse_args()
    sample(args)


def sample(args):
    timer = time.time()
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, infer=True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:

            def same_prefix(splitted, real, prefix_len=3):
                for i in range(len(real)):
                    try:
                        if splitted[i][:prefix_len].lower() != real[i][:prefix_len].lower():
                            return False
                    except IndexError:
                        return False
                return True

            saver.restore(sess, ckpt.model_checkpoint_path)
            k = 0
            count = 0
            random.seed()
            randlist = np.zeros(10)
            for i in range(10):
                # need to switch 66206 to sum(1 for line in f)
                randlist[i] = random.randint(0, 66206)
            with open('data/compounds.txt', encoding='utf-8') as f:
                for line in f:
                    if count in randlist:
                        words = line.split('\t')
                        words[-1] = words[-1][:-1]
                        if not ' ' in words[0] and not '-' in words[0]:
                            splitted = model.smash(sess, vocab, words[0].lower())
                            print(splitted)
                            print(words[1:])
                            if same_prefix(splitted, words[1:]):
                                k += 1
                    count += 1
            print(k / 10)

    print(time.time() - timer, 'seconds for test')

if __name__ == '__main__':
    main()
