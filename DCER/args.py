import argparse


def get_parser():
    # Training settings
    # 创建解析器
    parser = argparse.ArgumentParser(description='model parameters')
    # data
    parser.add_argument('--glove', default='../dataset/new.txt')
    # 添加参数
    # --dataset: amazon-ele/yelp
    parser.add_argument('--dataset', type=str, default='yelp', metavar='N', help='dataset')
    parser.add_argument('--ourl', type=str, default='all', metavar='N', help='')
    parser.add_argument('--murl', type=str, default='all', metavar='N', help='')
    parser.add_argument('--turl', type=str, default='all', metavar='N', help='')
    parser.add_argument('--psychological', type=str, default='all', metavar='N', help='')
    parser.add_argument('--channel', type=str, default='all', metavar='N', help='')
    parser.add_argument('--lamda', type=str, default='0.5', metavar='N', help='')

    # @config --batch_size/embed_dim
    # amazon-ele: 32/64(default)   40/64(best)  128/64(now)
    # yelp: 60/64
    #       26/128
    parser.add_argument('--batch_size', type=int , default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    # @config --drop_out
    # amazon-ele: 0.5(default)
    parser.add_argument('--drop_out', default=0.3, type=int)
    parser.add_argument('--word_dim', type=int, default=100, metavar='N', help='word embedding size')
    # @config --vocab_size_u
    # amazon-ele: 40291
    # yelp: 47680
    parser.add_argument('--vocab_size_u', type=int, default=40291, metavar='N', help='vocab size')
    parser.add_argument('--vocab_size_v', type=int, default=40291, metavar='N', help='vocab size')
    parser.add_argument('--filter_sizes', default='1,2,3', type=str)
    parser.add_argument('--num_filters', default=100, type=int)
    # TODO @config --seq_len
    # amazon-ele: 346
    # yelp: 206
    parser.add_argument('--seq_len', default=346, type=int)
    return parser
