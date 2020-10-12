from torch.nn import Parameter
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from GCN_RE.GAT import GraphAttentionLayer, SpGraphAttentionLayer
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()

NAMESPACE = 'GCN'

class TextCNN(nn.Module):
    def __init__(self, output_size, input_dim):
        super(TextCNN, self).__init__()

        self.filter_size = [3,4,5]
        self.filter_num = 64
        self.channel_num = 1
        self.input_dim = input_dim
        self.class_num = output_size

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.channel_num, self.filter_num,
                       (size, self.input_dim)) for size in self.filter_size])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(self.filter_size) * self.filter_num, self.class_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_conv = [F.relu(convd(x)).squeeze(3) for convd in self.convs]
        x_pool = [F.avg_pool1d(item, item.size(2)).squeeze(2) for item in x_conv]
        x_stack = torch.cat(x_pool, 1)
        x_drop = self.dropout(x_stack)
        activation = torch.nn.ReLU()
        x_relu = activation(x_drop)
        logits = self.fc(x_relu)
        return logits

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super(GraphConvolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(0.5)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, batch_size):
        result = []
        for i in range(batch_size):
            support = torch.matmul(input[i], self.weight)
            output = torch.matmul(adj[i], support)
            # print(output.shape)
            if self.bias is not None:
                output =  output + self.bias
            activation = torch.nn.LeakyReLU()
            result.append(activation(self.dropout(output)))
        # result = np.array(result)
        return result

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, num_layers):
        super(BiRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.output = nn.Linear(self.hidden_size*2, self.output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.output(out)
        activation = torch.nn.LeakyReLU()
        out = activation(out)
        return out

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, batch_size):
        result = []
        for i in range(batch_size):
            x_tensor = x[i]
            adj_tensor = adj[i]
            x_tensor = F.dropout(x_tensor, self.dropout, training=self.training)
            x_tensor = torch.cat([att(x_tensor, adj_tensor) for att in self.attentions], dim=1)
            x_tensor = F.dropout(x_tensor, self.dropout, training=self.training)
            x_tensor = F.leaky_relu(self.out_att(x_tensor, adj_tensor))
            result.append(x_tensor)
        return result

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj, batch_size):
        result = []
        for i in range(batch_size):
            x_tensor = x[i]
            adj_tensor = adj[i]
            # x_tensor = F.dropout(x_tensor, self.dropout, training=self.training)
            x_tensor = torch.cat([att(x_tensor, adj_tensor) for att in self.attentions], dim=1)
            # x_tensor = F.dropout(x_tensor, self.dropout, training=self.training)
            x_tensor = F.leaky_relu(self.out_att(x_tensor, adj_tensor))
            result.append(x_tensor)
        return result

class GCNReModel(nn.Module):
    _stack_dimension = 2
    _embedding_size = 256
    _output_dim_lstm = 256
    _memory_dim = 256
    _vocab_size = 300
    _hidden_layer1_size = 200
    _hidden_layer2_size = 200
    _output_size = 42
    _alpha = 0.2
    maxlength = 256
    _global_dim = 20
    _dropout = 0.5

    def __init__(self, args):
        super(GCNReModel, self).__init__()
        self.args = args
        self._output_size = self.args.output_size
        # GraphConvolution
        self.Bi_LSTM = BiRNN(input_size=self._vocab_size, hidden_size=self._memory_dim,
                             output_dim=self._output_dim_lstm, num_layers=2)
        # self.DGC = GraphConvolution(self._output_dim_lstm, self._hidden_layer1_size)
        # self.DGC_1 = GraphConvolution(self._hidden_layer1_size, self._hidden_layer1_size)
        # self.EGC = GraphConvolution(self._hidden_layer1_size, self._hidden_layer2_size)
        # self.EGC_1 = GraphConvolution(self._hidden_layer2_size, self._hidden_layer2_size)
        self.Text_cnn = TextCNN(output_size = self._output_size, input_dim=self._hidden_layer2_size)
        self.fc = nn.Linear(self.maxlength, self._global_dim)
        self.fc1 = nn.Linear(self._output_dim_lstm, self._hidden_layer2_size)
        self.DGC = GraphConvolution(self._output_dim_lstm, self._hidden_layer1_size)
        #self.EGC = GraphConvolution(self._hidden_layer1_size, self._hidden_layer2_size)
        #self.DGC_att = GAT(nfeat=self._output_dim_lstm, nhid=8, nheads=8, nclass=self._hidden_layer1_size,alpha=self._alpha, dropout=0.5)
        self.EGC_att = GAT(nfeat=self._hidden_layer1_size, nhid=8, nheads=8, nclass=self._hidden_layer2_size,
                                           alpha=self._alpha, dropout=0.5)

    def forward(self, x, aj_matrix_1, aj_matrix_2, subj_start, subj_end, obj_start, obj_end, batch_size):
        # print(x.shape)
        LSTM_out = self.Bi_LSTM(x)
        # DGC_out = self.DGC(LSTM_out, aj_matrix_1, batch_size)
        # DGC_out_1 = self.DGC_1(DGC_out, aj_matrix_1, batch_size)
        # EGC_out = self.EGC(DGC_out_1, aj_matrix_2, batch_size)
        # EGC_out_1 = self.EGC_1(EGC_out, aj_matrix_2, batch_size)

        DGC_out = self.DGC(LSTM_out, aj_matrix_1, batch_size)
        EGC_out = self.EGC_att(DGC_out, aj_matrix_2, batch_size)
        prediction = []

        for i in range(len(EGC_out)):
            subj = EGC_out[i][subj_start:subj_end+1, :]
            obj = EGC_out[i][obj_start:obj_end+1, :]
            entity_pair = torch.cat((subj, obj), 0)

            global_feature = LSTM_out[i]
            global_feature = torch.transpose(self.fc(torch.transpose(global_feature,0,1)),0,1)
            global_feature = self.fc1(global_feature)
            entity_pair_full = torch.cat((entity_pair, global_feature), 0)
            prediction.append(entity_pair_full)
        prediction_t = torch.stack(prediction, dim = 0)
        logits = self.Text_cnn(prediction_t)
        return logits

class Train(object):
    def __init__(self, maxlength, args):
        self.maxlength = maxlength
        self.batch_size = 100
        self.model = GCNReModel(args).cuda()
        self.args = args
        # self.pos_weight = torch.from_numpy(np.array(pos_weight, dtype=np.float64)).float().cuda()
        # print(pos_weight)
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def __train(self, A_GCN_l1,A_GCN_l2, X, y, subj_start, subj_end, obj_start, obj_end):
        Aj_matrix_1 = np.array([item for item in A_GCN_l1])
        Aj_matrix_2 = np.array([item for item in A_GCN_l2])

        # print(len(X))
        X_array = np.array(X)
        y_array = np.squeeze(np.array(y))
        # y_array = np.transpose(y_array, (1, 0, 2))
        self.batch_size = X_array.shape[0]
        X_array = np.squeeze(X_array)

        # print(Aj_matrix_1.shape)
        Aj_matrix_1 = torch.from_numpy(Aj_matrix_1).float().cuda()
        Aj_matrix_2 = torch.from_numpy(Aj_matrix_2).float().cuda()
        X2 = torch.from_numpy(X_array).float().cuda()
        y_array = torch.from_numpy(y_array).float().cuda()

        Aj_matrix_1_gen = []
        for item in Aj_matrix_1:
            Aj_matrix_1_gen.append(self.gen_adj(item))

        Aj_matrix_2_gen = []
        for item in Aj_matrix_2:
            Aj_matrix_2_gen.append(self.gen_adj(item))

        if len(X) == 1:
            X2 = torch.unsqueeze(X2, 0)
            y_array = torch.unsqueeze(y_array, 0)

        subj_start = int(subj_start[0])
        subj_end = int(subj_end[0])
        obj_start = int(obj_start[0])
        obj_end = int(obj_end[0])

        self.optimizer.zero_grad()

        prediction = self.model(X2, Aj_matrix_1_gen, Aj_matrix_2_gen, subj_start, subj_end,
                                obj_start, obj_end, self.batch_size,)
        # criterion = nn.CrossEntropyLoss().cuda()
        # loss = criterion(prediction, torch.argmax(y_array, 1))

        criterion = nn.BCEWithLogitsLoss().cuda()
        loss = criterion(prediction, y_array)

        # print(torch.softmax(prediction, 1))
        # print(y_array)
        # print(torch.argmax(torch.softmax(prediction, 1), 1))
        # print(torch.argmax(y_array, 1))
        loss.backward()
        self.optimizer.step()

        target = torch.argmax(prediction, 1)
        correct = 0
        correct += (target == torch.argmax(y_array, 1)).sum().float()

        return correct, loss

    def train(self, data):
        correct, loss = self.__train([data[i][0] for i in range(len(data))],
                            [data[i][1] for i in range(len(data))],
                            [data[i][2] for i in range(len(data))],
                            [data[i][3] for i in range(len(data))],
                            [data[i][4] for i in range(len(data))],
                            [data[i][5] for i in range(len(data))],
                            [data[i][6] for i in range(len(data))],
                            [data[i][7] for i in range(len(data))],
                           )


        # print("acc: " + str(acc))
        return correct, loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def gen_adj(self, A):
        # print(A)
        # print(A.sum(1))
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(D, A), D)
        return adj

class Eval(object):
    def __init__(self, maxlength, model_path, args):
        self.maxlength = maxlength
        self.batch_size = 100
        self.model = GCNReModel(args=args).cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def _predict(self, A_GCN_l1,A_GCN_l2, X, y, subj_start, subj_end, obj_start, obj_end, RE_filename, threshold):
        Aj_matrix_1 = np.array([item for item in A_GCN_l1])
        Aj_matrix_2 = np.array([item for item in A_GCN_l2])
        correct = 0
        X_array = np.array(X)
        y_array = np.squeeze(np.array(y))
        # y_array = np.transpose(y_array, (1, 0, 2))
        self.batch_size = X_array.shape[0]
        X_array = np.squeeze(X_array)

        Aj_matrix_1 = torch.from_numpy(Aj_matrix_1).float().cuda()
        Aj_matrix_2 = torch.from_numpy(Aj_matrix_2).float().cuda()
        X2 = torch.from_numpy(X_array).float().cuda()
        y_array = torch.from_numpy(y_array).float().cuda()

        Aj_matrix_1_gen = []
        for item in Aj_matrix_1:
            Aj_matrix_1_gen.append(self.gen_adj(item))

        Aj_matrix_2_gen = []
        for item in Aj_matrix_2:
            Aj_matrix_2_gen.append(self.gen_adj(item))

        if len(X) == 1:
            X2 = torch.unsqueeze(X2, 0)
            y_array = torch.unsqueeze(y_array, 0)

        subj_start = int(subj_start[0])
        subj_end = int(subj_end[0])
        obj_start = int(obj_start[0])
        obj_end = int(obj_end[0])

        prediction = self.model(X2, Aj_matrix_1_gen, Aj_matrix_2_gen, subj_start, subj_end, obj_start, obj_end, self.batch_size)

        prediction_class = torch.sigmoid(prediction)
        TP, FN, FP = self.getPRF(prediction_class, target=y_array, threshold=threshold)

        return TP, FN ,FP
    def predict(self, data, RE_filename, threshold):
        # outputs = np.array(self._predict([A_fw], [A_bw], [X], [value_matrix], [A_fw_dig], [A_bw_dig]))
        TP, FN ,FP = self._predict([data[i][0] for i in range(len(data))],
                            [data[i][1] for i in range(len(data))],
                            [data[i][2] for i in range(len(data))],
                            [data[i][3] for i in range(len(data))],
                            [data[i][4] for i in range(len(data))],
                            [data[i][5] for i in range(len(data))],
                            [data[i][6] for i in range(len(data))],
                            [data[i][7] for i in range(len(data))],
                                RE_filename,
                                threshold,
                           )
        return TP, FN ,FP

    def gen_adj(self, A):
        # print(A)
        # print(A.sum(1))
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(D, A), D)
        return adj

    def getPRF(self, predict, target, threshold):
        TP = 0.0
        FN = 0.0
        FP = 0.0
        for i in range(predict.shape[0]):
            predict_vec  = predict[i]
            target_vec = target[i]
            # print(predict_vec)
            # print(target_vec)
            for j in range(len(predict_vec)):
                if predict_vec[j] >= threshold:
                    if target_vec[j] == 1:
                        TP += 1
                    else:
                        FP += 1
                if predict_vec[j] < threshold:
                    if target_vec[j] == 1:
                        FN += 1
        # presicion = float(TP)/float(TP+FP)
        # recall = float(TP)/float(TP+FN)
        # F1 = 2*presicion*recall/(presicion+recall)
        return TP, FN ,FP
