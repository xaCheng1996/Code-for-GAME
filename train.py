from GCN_RE import GCNRE
import sys
import os
import argparse

args_parser = argparse.ArgumentParser(description='GCN for RE')
#learning parameters
args_parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate[default:0.001]')
args_parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]')
args_parser.add_argument('-batch-size', type=int, default=16, help='batch size for training [default: 100]')
args_parser.add_argument('-save-dir', type=str, default='./data', help='where to save the snapshot')
args_parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
args_parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
args_parser.add_argument('-maxlength', type=int, default=256, help='length of sentences which is padded')
# data
args_parser.add_argument('-dataset', type=int, default=1, help='the chose of dataset, 0 for TACRED, 1  NYT')
args_parser.add_argument('-output_size', type=int, default=24, help='the chose of dataset, 42 for TACRED, 10 for SemEval, 24 for NYT')
args_parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# model
args_parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
args_parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
args_parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
# device
args_parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
args_parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
args_parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
args_parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
args_parser.add_argument('-test', action='store_true', default=False, help='train or test')

args = args_parser.parse_args()

if __name__ == '__main__':
    data_name = ['TacRED', 'SemEval', 'NYT']
    data_train = ["./train_full.json", "./data/nyt/raw_nyt/train_entity.json"]
    model_save = ['./data/TACRED/eval', './data/SemEval2010_task8/eval', './data/nyt/eval']
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    GCNRE.train_and_save(dataset=data_train[args.dataset], saving_dir=model_save[args.dataset],
                         data_name = data_name[args.dataset],
                         epochs=args.epochs, bucket_size = args.batch_size, args=args)
