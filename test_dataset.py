import os
from GCN_RE import GCNRE

if __name__ == '__main__':
        checkpoint = 14
        data_name = ['TacRED']
        RE_file = './data/TACRED/eval/gcn-re-'+str(checkpoint)+'.tf'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        RE = GCNRE()
        RE.test(RE_filename=RE_file, dataset='./test.json', data_name = data_name[0], threshold=0.6)
