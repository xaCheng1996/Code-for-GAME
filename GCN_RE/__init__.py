import numpy as np
import pickle
import random
import sys
import logging

import GCN_RE
from GCN_RE.GCN_model import Eval
import GCN_RE.utils as utils


class GCNRE:
    _logger = logging.getLogger(__name__)
    def test(self, RE_filename, checkpoint, dataset, data_name, threshold, batch_size, args):
        '''
        The system tests the current NER model against a text in the CONLL format.

        :param dataset: the filename of a text in the CONLL format
        :return: None, the function prints precision, recall and chunck F1
        '''
        sentences = utils.auxiliary.get_all_sentence(dataset, data_name)
        maxlength = 256
        # data = utils.auxiliary.get_data_from_sentences(sentences, maxlength)
        wt = open('./RE_result.out', 'r', encoding='utf-8')
        for i in range(5, checkpoint):
            RE_file = RE_filename + '/gcn-re-param-' + str(i) + '.pkl'
            self.model = Eval(batch_size, RE_file, args)
            P,R,F = utils.testing.get_gcn_results(self.model, sentences, maxlength, batch_size, RE_file, threshold=threshold)
            print('The checkpoint {}\'s P R F is {:.4f}:{:.4f}:{:.4f}:'.format(RE_file, P, R, F))
#            wt.write('The checkpoint {}\'s P R F is {:.4f}:{:.4f}:{:.4f}:\n'.format(RE_filename, P, R, F))
        # print('recall:', recall)
        # print('F1:', f1)

    @staticmethod
    def train_and_save(dataset, saving_dir, data_name, epochs, bucket_size, args):
        '''
        :param dataset: A file use as a training.
        :param saving_dir: The directory where to save the results
        :param epochs: The number of epochs to use in the training
        :param bucket_size: The batch size of the training.
        :return: An instance of this class
        '''
        print('Training the system according to the dataset ', dataset)
        return utils.training.train_and_save(dataset, saving_dir, data_name, epochs, bucket_size, args)
