# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/22 15:49'

import numpy
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from sklearn import metrics
from torch import nn
from bucket_iterator import BucketIterator
from data import makeBert_datalodaer
from data.make_datalodaer import DatesetReader

from metrics import evaluate, evaluate_MT
from model.ATGRU.ATGRU import ATGRU
from model.ATJSS.model import BaseModel
# from model.BERT.SKEP import SkepForClassification
from model.BERT.base_BERT import BertForClassification
from model.BiLSTM.BiLSTM import BiLSTM
from model.GCAE.myGCAE import GCAE
from model.TAN.TAN import LSTM_TAN
from model.kimCNN.KimCNN import KimCNN
from options import prepare_train_args
from utils.logger import Logger
from loss import loss_function
from utils.GCN_preData import PrepareData
from utils.get_embedding import get_wordvec
import numpy as np
np.set_printoptions(threshold=np.inf)

# from model.kimCNN.Textcnn import textCNN
from model.MCNN_MT.MCNN_MT import Multi_Channel_CNN
from model.CO_LSTM.CO_LSTM import CoLSTM
from model.ABCDM.ABCDM import ABCDM
from model.BiLSTM.DC_BiLSTM import T_DAN

device = "cuda:0"
use_Bert = False
is_MT = True
is_graph = False

class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        # train_loader, val_loader, word_vectors, X_lang, Target = make_dataloader(batch_size=100, debug=False, is_train=True)

        # val_loader, word_vectors, X_lang, Target  = make_dataloader(batch_size=100, debug=False, is_train=False)
        if use_Bert == False:
            print("使用glove获得词向量===========")
            covid_dataset = DatesetReader(dataset="covid-19-tweet", embed_dim=300)
        else:
            print("使用bert来embedding=========")
            covid_dataset = makeBert_datalodaer.DatesetReader(dataset="covid-19-tweet")

        print("已得到dataset，正在准备dataloader.....")
        #准备two-step数据集
        for item in covid_dataset.train_data:
            if item['stance'] == 1 or item['stance'] == 0:
                item['subject'] = 1
            else:
                item['subject'] = 0

        for item in covid_dataset.test_data:
            if item['stance'] == 1 or item['stance'] == 0:
                item['subject'] = 1
            else:
                item['subject'] = 0

        for item in covid_dataset.train_data:
            if item['stance'] == 1:
                item['pn'] = 1
            else:
                item['pn'] = 0

        for item in covid_dataset.test_data:
            if item['stance'] == 1:
                item['pn'] = 1
            else:
                item['pn'] = 0

        self.train_data_loader = BucketIterator(data=covid_dataset.train_data, batch_size=64,
                                                shuffle=True)
        self.test_data_loader = BucketIterator(data=covid_dataset.test_data, batch_size=64,
                                               shuffle=False)



        if is_graph:
            predata = PrepareData()
            self.feature = predata.features
            self.adj = predata.adj

        # for i, sample_batched in enumerate(self.train_data_loader):
            # print("+++++max_len = ",sample_batched['max_len'])
            # print("##########",sample_batched['text'])



        # self.model = torch.nn.DataParallel(self.model)
        # self.model = BiLSTM(input_size=len(covid_dataset.embedding_matrix), embedding_dim=300, hidden_size=512, weights=covid_dataset.embedding_matrix, dropout=0.5)
        # self.model = KimCNN(input_size=len(covid_dataset.embedding_matrix), embedding_dim=300, n_filters=16, kernel_size=[2,3,4], weights=covid_dataset.embedding_matrix, net_dropout=0.5)
        # self.model = LSTM_TAN(embedding_dim=300, hidden_dim=512, vocab_size=len(covid_dataset.embedding_matrix), n_targets=3,embedding_matrix=covid_dataset.embedding_matrix,dropout = 0.5)
        # self.model = ATGRU(linear_size=len(covid_dataset.embedding_matrix), lstm_hidden_size=512,embedding_dim=300, embedding_matrix=covid_dataset.embedding_matrix, net_dropout=0.5)
        # self.model = GCAE(input_size=len(covid_dataset.embedding_matrix), embedding_dim=300, n_filters=16, kernel_size=[2, 3, 4],
        #                     weights=covid_dataset.embedding_matrix, net_dropout=0.5)
        # self.model = BertForClassification(args.bert_path,args.num_classes)  #纯BERT
        # self.model = bertLSTM(args.bert_path,args.num_classes)
        # self.model = Multi_Channel_CNN(len(covid_dataset.embedding_matrix),au_vocab_size_list=[len(get_wordvec("100_tag_embedding_matrix.pkl")),len(get_wordvec("100_position_embedding_matrix.pkl"))],
        #                                embedding_dim=300,hidden_dim=512,n_filters=100,kernel_size=[2,3,4],embed_weights=covid_dataset.embedding_matrix,
        #                                au_weight=[get_wordvec("100_tag_embedding_matrix.pkl"),get_wordvec("100_position_embedding_matrix.pkl")])
        # self.model = CoLSTM(input_size=len(covid_dataset.embedding_matrix),n_filters=16,kernel_size=[2,3,4],embedding_dim=300,hidden_size=512,weights=covid_dataset.embedding_matrix)
        # self.model = ABCDM(input_size=len(covid_dataset.embedding_matrix),embedding_dim=300, hidden_size=512, n_filters=32,kernel_size=[2,3,4],weights=covid_dataset.embedding_matrix)
        # self.model = SkepForClassification(args.bert_path, args.num_classes)
        self.model = T_DAN(input_size=len(covid_dataset.embedding_matrix), embedding_dim=300, hidden_size=512, weights=covid_dataset.embedding_matrix)

        #多任务模型：
        #ATJSS:
        # self.model = BaseModel(linear_size=len(covid_dataset.embedding_matrix),hidden_size=512,embedding_dim=300, embedding_matrix=covid_dataset.embedding_matrix,dropout=0.5)

        #图GNN：
        # self.model = GCN(nfeat=predata.nfeat_dim, nhid=200, nclass=predata.nclass, dropout=0.5)


        self.model = [self.model.to(device), self.model.to(device)]
        # self.model = self.model.to(device)



        self.params= [self.model[0].parameters(),self.model[1].parameters()]


        # # 启动 batchNormal 和 dropout
        # param_optimizer = list(self.model.named_parameters())
        # # 不需要衰减的参数  #衰减：（修正的L2正则化）
        # no_decay = ['bias', 'LayerNorm.bias', 'Layerweight']
        #
        # # 指定哪些权重更新，哪些权重不更新
        # optimizmer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        #     # 遍历所有参数，如果参数名字里有no_decay的元素则取出元素
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #     # 遍历所有参数，如果参数名字里没有no_decay的元素则取出元素
        # ]

        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
        # self.optimizer = torch.optim.Adam(optimizmer_grouped_parameters, self.args.lr,
        #                                   betas=(self.args.momentum, self.args.beta),
        #                                   weight_decay=self.args.weight_decay)

        self.opts = [torch.optim.Adam(self.params[0], lr=self.args.lr, weight_decay=self.args.weight_decay,
                                       betas=(self.args.momentum, self.args.beta)),
                     torch.optim.Adam(self.params[1], lr=self.args.lr, weight_decay=self.args.weight_decay,
                                       betas=(self.args.momentum, self.args.beta))]
        # self.schedulers = []

        self.opt_1, self.opt_2 = self.opts

        self.inputs_cols = ['text_indices','attention_mask','target_indices','tag_indices','position_indices']
        self.criterion = torch.nn.CrossEntropyLoss()


    '''训练入口'''
    def train(self):
        self.best_acc = 0
        self.best_f1 = 0
        self.best_mif1 = 0
        self.best_avf1 = 0
        self.best_pr = 0
        self.best_re = 0
        self.best_au_f1 = 0
        self.best_au_mif1 = 0
        self.best_au_avf1 = 0
        self.best_au_acc = 0
        self.best_au_pr = 0
        self.best_au_re = 0
        # self.p_k = [1/2,1/2]
        # self.rho = 1.2
        # self.eta_p = 0.5
        # self.w = [0.1201, 0.2933]
        self.tasks = ["Stance", "Sentiment"]

        for epoch in range(self.args.epochs):
            # train for one epoch
            print("Begin training....")

            self.train_per_epoch(epoch)
            #验证
            self.val_per_epoch(epoch)
            # self.logger.save_curves(epoch)
            # self.logger.save_check_point(self.model, epoch)
        print("\n"
              "+++++++++++After all epoch, best_Accuracy : {}, precision : {}, recall: {}, f1 : {}, micro_f1 : {}, avg_f1 : {}".format(self.best_acc,
                                                                                                                        self.best_pr,self.best_re,
                                                                                                                        self.best_f1,self.best_mif1,
                                                                                                                                       self.best_avf1))
        if is_MT == True:
            print("au_task_acc :{}, au_task_pr :{}, au_task_re :{} ,au_task_f1 : {}, au_task_micro_f1 : {}, au_task_avg_f1 : {}".format(self.best_au_acc, self.best_au_pr, self.best_au_re,self.best_au_f1, self.best_au_mif1, self.best_au_avf1))

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model[0].train()
        self.model[1].train()

        for i, sample_batched in enumerate(self.train_data_loader):
            # losses = []

            # x, label, pred = self.step(sample_batched)
            x = [sample_batched[col].cuda() for col in self.inputs_cols]

            # print(x)
            t1_label = sample_batched['subject'].cuda()
            t1_pred = self.model[0](x)
            # compute loss
            # print(pred)
            # print(label)
            metrics = self.compute_metrics(t1_pred, t1_label, is_train=True)

            # get the item for backward
            loss = metrics['train/l1']

            # compute gradient and do Adam step
            self.opt_1.zero_grad()
            loss.backward()
            self.opt_1.step()


            t2_label = sample_batched['np'].cuda()
            t2_pred = self.model[1](x)  # 通用

            metrics = self.compute_metrics(t2_pred, t2_label, is_train=True)

            # get the item for backward
            loss = metrics['train/l1']

            # compute gradient and do Adam step
            self.opt_2.zero_grad()
            loss.backward()
            self.opt_2.step()


            if i % 10 == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))
                #每次都有bachsize个


    def val_per_epoch(self, epoch):
        # self.model[0].eval()
        # self.model[1].eval()
        #
        # val_acc, val_loss,val_pr, val_re, val_f1, val_mif1 = evaluate(model=self.model, criterion=nn.CrossEntropyLoss(), dataloader=self.test_data_loader,
        #                                           device=device) #task_type=args.task_type target = self.Target
        # print(
        #     "Epoch {} complete! Accuracy : {}, Loss : {}, precision : {}, recall: {}, f1 : {}, micro_f1 : {}, avg_f1 : {}".format(epoch, val_acc, val_loss,
        #                                                                   val_pr, val_re
        #                                                                   , val_f1, val_mif1,
        #                                                                   (val_f1 + val_mif1) / 2))
        #
        #
        # if val_acc > self.best_acc:
        #     print("Best validation accuracy improved from {} to {}, saving model...".format(self.best_acc, val_acc))
        #     self.best_acc = val_acc
        #     self.best_f1 = val_f1
        #     self.best_mif1 = val_mif1
        #     self.best_avf1 = (self.best_f1 + self.best_mif1) / 2
        #     self.best_pr = val_pr
        #     self.best_re = val_re
        with torch.no_grad():
            for i in range(len(self.model)):
                self.model[i].eval()
            accs = []
            losses_var = []
            acc = f1 = mif1 = re = pr = 0
        # self.model.eval()


        # st_n_pre, st_n_labels = None, None
        # senti_n_pre, senti_n_labels = None, None
        n_pre = None
        n_labels = None
        #只计算主任务的acc等

        for i, sample_batched in enumerate(self.test_data_loader):
            x = [sample_batched[col].cuda() for col in self.inputs_cols]
            subject = sample_batched['subject'].to(device)
            stance = sample_batched['stance'].to(device)
            y1_pred = self.model[0](x)
            if y1_pred == 1: #是主观，再判断积极消极
                y2_pred = self.model[1](x)
            else:
                y2_pred = 2

            # loss1 = self.criterion(y, y[ind])
            # test_loss += loss.item()
            # n_acc += y_.argmax(1).eq(y[ind]).sum()
            # n_all += y[ind].shape[0]

            if (n_pre == None):
                n_pre = torch.argmax(y2_pred, 1)
                n_labels = stance
                # senti_n_pre = torch.argmax(senti_pred, 1)
                # senti_n_labels = sentiment
            else:
                n_pre = torch.cat((n_pre, y2_pred), dim=0)
                n_labels = torch.cat((n_labels, stance), dim=0)
                # senti_n_pre = torch.cat((senti_n_pre, torch.argmax(senti_pred, 1)), dim=0)
                # senti_n_labels = torch.cat((senti_n_labels, sentiment), dim=0)



            # acc = n_acc / float(n_all)
            # accs.append(acc)
            # losses_var.append(test_loss)
            # print('loss/test/{t} epoch+++++++++', test_loss, epoch)
            # print('acc/test/{t} epoch+++++++++', acc, epoch)

        acc = metrics.accuracy_score(n_labels.cpu(), n_pre.cpu())
        f1 = metrics.f1_score(n_labels.cpu(), n_pre.cpu(), labels=[0, 1], average='macro')
        mif1 = metrics.f1_score(n_labels.cpu(), n_pre.cpu(), labels=[0, 1], average='micro')
        re = metrics.recall_score(n_labels.cpu(), n_pre.cpu(), labels=[0, 1], average='macro')
        pr = metrics.precision_score(n_labels.cpu(), n_pre.cpu(), labels=[0, 1], average="macro")


        # senti_f1 = metrics.f1_score(senti_n_labels.cpu(), senti_n_pre.cpu(), labels=[0, 1], average='macro')
        # senti_mif1 = metrics.f1_score(senti_n_labels.cpu(), senti_n_pre.cpu(), labels=[0, 1], average='micro')
        print(
            "Epoch {} complete! Accuracy : {}, precision : {}, recall: {}, f1 : {}, micro_f1 : {}, avg_f1 : {}\n".format(epoch,acc,
                                                                          pr, re
                                                                          , f1, mif1,
                                                                          (f1 + mif1) / 2))
        # if acc >= 0.64 and acc <= 0.66:
        #     breakpoint()



        if acc > self.best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(self.best_acc, acc))
            self.best_acc = acc
            self.best_f1 = f1
            self.best_mif1 = mif1
            self.best_avf1 = (self.best_f1 + self.best_mif1) / 2
            self.best_pr = pr
            self.best_re = re



    def compute_metrics(self, pred, gt, is_train):
        # you can call functions in metrics.py
        loss_fuc = torch.nn.CrossEntropyLoss()
        # l1 = (pred - gt).abs().mean()
        l1 = loss_fuc(pred, gt)
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + 'l1': l1
        }
        return metrics

    def gen_imgs_to_write(self, x, pred, label, is_train):
        # override this method according to your visualization
        prefix = 'train/' if is_train else 'val/'
        return {
            prefix + 'x': x[0],
            prefix + 'pred': pred[0],
            prefix + 'label': label[0]
        }

    def compute_loss(self, pred, gt):
        if self.args.loss == 'l1':
            loss = (pred - gt).abs().mean()
        elif self.args.loss == 'ce':
            loss = torch.nn.functional.cross_entropy(pred, gt)
        else:
            loss = torch.nn.functional.mse_loss(pred, gt)
        return loss




def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
