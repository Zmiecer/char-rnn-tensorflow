import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

import argparse
import time
from six.moves import cPickle

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, default='data/input',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--experiment_log', type=str, default='experiments.txt',
                        help='file to store experiment parameters (appended)')
    parser.add_argument('--loss_log', type=str, default='test_loss.txt',
                        help='file to store test loss progress (appended)')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=30,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=10000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='learning rate lower bound (stopping condition)')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path.
                       Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths,
                                                  be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--device', type=str, default='/gpu:0',
                        help='device')
    args = parser.parse_args()
    train(args)
    with open(args.experiment_log, mode='a') as logfile:
        print('\nData: {}, {} layers, hidden state size {}, cell type {}'.format(
            args.data_dir, args.num_layers, args.rnn_size, args.model
        ), file=logfile)


def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),\
            "config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),\
            "chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],\
                "Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars == data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)
    model = Model(args)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
            os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))
        )
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        prev_loss = np.Infinity
        av_train_loss = 0
        min_test_loss = np.Infinity

        with open(args.loss_log, mode='a') as loss_log:
            print("\nData: {}, number of layers: {}, hidden state size: {}".format(
                args.data_dir, args.num_layers, args.rnn_size
            ), file=loss_log)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            state = sess.run(model.initial_state)
            b = 1
            start = time.time()

            data_loader.next_epoch()
            for x, y, word_lengths in data_loader.iterbatches('train'):
                feed = {model.input_data: x, model.targets: y, model.word_len: word_lengths}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h

                # instrument for tensorboard
                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summ, e * data_loader.num_batches + b)
                writer.flush()

                av_train_loss += train_loss

                end = time.time()
                print("{}/{} (file {}/{}, epoch {}/{}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                    b - data_loader.num_batches*data_loader.file_number, data_loader.num_batches,
                    data_loader.file_number, data_loader.num_train_files - 1,
                    e, args.num_epochs - 1, train_loss, end - start
                ))
                b += 1

                start = time.time()

            av_train_loss /= b

            # test
            av_test_loss = 0
            t = 1
            start = time.time()
            for x_test, y_test, word_lengths in data_loader.iterbatches('test'):
                feed = {model.input_data: x_test, model.targets: y_test, model.word_len: word_lengths}
                test_loss = sess.run([model.cost], feed)
                test_loss = test_loss[0]
                av_test_loss += test_loss
                end = time.time()
                print("{}/{} (file {}/{}, epoch {}/{}), test_loss = {:.3f}, time/batch = {:.3f}".format(
                    t - data_loader.num_batches*(data_loader.file_number - data_loader.num_train_files),
                    data_loader.num_batches,
                    data_loader.file_number - data_loader.num_train_files - 1, data_loader.num_test_files - 1,
                    e, args.num_epochs - 1,
                    test_loss, end - start
                ))
                t += 1
                start = time.time()

            av_test_loss /= t

            with open(args.loss_log, mode='a') as loss_log:
                print("Epoch {}, learning_rate = {}, train_loss = {}, test_loss = {}"
                      .format(e, args.learning_rate, av_train_loss, av_test_loss), file=loss_log)

            if av_test_loss >= prev_loss:
                args.learning_rate /= 2
                print(args.learning_rate)
            if av_test_loss < min_test_loss:
                checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path,
                           global_step=e * data_loader.num_batches * data_loader.num_train_files)
                print("model saved to {}".format(checkpoint_path))
                min_test_loss = av_test_loss
            prev_loss = av_test_loss

            print("epoch {} ended".format(e))
            if args.learning_rate < args.min_lr:
                print("Minimum reached", args.learning_rate, "<", args.min_lr)
                break


if __name__ == '__main__':
    main()
