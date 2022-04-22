import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch import nn
from bucket_iterator import BucketIterator
from data import makeBert_datalodaer
from data.ABSA import makeBert_datalodaer_ABSA
from data.ABSA.bucket_iterator_ABSA import BucketIterator_ABSA
from data.make_datalodaer import DatesetReader
from evaluation import get_metric

from model.DCRAN.model_bert import DCRAN_BERT
from options import prepare_train_args
from utils.logger import Logger
from loss import loss_function
from utils.GCN_preData import PrepareData
from utils.get_embedding import get_wordvec
import numpy as np
np.set_printoptions(threshold=np.inf)


device = "cuda:1"
use_Bert = True
is_MT = False
is_graph = False

class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)
        print(('Device is :', device))
        # train_loader, val_loader, word_vectors, X_lang, Target = make_dataloader(batch_size=100, debug=False, is_train=True)

        # val_loader, word_vectors, X_lang, Target  = make_dataloader(batch_size=100, debug=False, is_train=False)
        if use_Bert == False:
            print("使用glove获得词向量===========")
            covid_dataset = DatesetReader(dataset="covid-19-tweet", embed_dim=300)
            # covid_dataset = DatesetReader(dataset="semEval2016", embed_dim=300)
        else:
            print("使用bert来embedding=========")
            # lap14_dataset = makeBert_datalodaer_ABSA.DatesetReader(dataset="lap14")
            res14_dataset = makeBert_datalodaer_ABSA.DatesetReader(dataset="res14")
            # covid_dataset = makeBert_datalodaer.DatesetReader(dataset="covid-19-tweet")
            # covid_dataset = makeBert_datalodaer.DatesetReader(dataset="semEval2016")

        print("已得到dataset，正在准备dataloader.....")
        # self.train_data_loader = BucketIterator(data=covid_dataset.train_data, batch_size=32,
        #                                         shuffle=True)
        # self.test_data_loader = BucketIterator(data=covid_dataset.test_data, batch_size=32,
        #                                        shuffle=False)
        self.train_data_loader = BucketIterator_ABSA(data=res14_dataset.train_data, batch_size=16,
                                                shuffle=True)
        self.dev_data_loader = BucketIterator_ABSA(data=res14_dataset.train_data, batch_size=16,
                                                shuffle=False)
        self.test_data_loader = BucketIterator_ABSA(data=res14_dataset.test_data, batch_size=16,
                                               shuffle=False)

        if is_graph:
            predata = PrepareData()
            self.feature = predata.features
            self.adj = predata.adj

        # for i, sample_batched in enumerate(self.train_data_loader):
            # print("+++++max_len = ",sample_batched['max_len'])
            # print("##########",sample_batched['text'])


        #ABSA模型
        self.model = DCRAN_BERT(args.bert_path, hidden_size=768)


        self.model = self.model.to(device)

        # 启动 batchNormal 和 dropout
        param_optimizer = list(self.model.named_parameters())
        # 不需要衰减的参数  #衰减：（修正的L2正则化）
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        no_decay = ['bias', 'LayerNorm.weight']

        # 指定哪些权重更新，哪些权重不更新
        optimizmer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            # 遍历所有参数，如果参数名字里有no_decay的元素则取出元素
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            # 遍历所有参数，如果参数名字里没有no_decay的元素则取出元素
        ]

        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
        self.optimizer = torch.optim.Adam(optimizmer_grouped_parameters, self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)

        if is_MT == True:
            self.opt_c, self.opt_g = [self.optimizer, self.optimizer]

        # self.inputs_cols = ['text_indices','attention_mask','target_indices','tag_indices','position_indices',"graph"]
        # self.inputs_cols = ['text_indices','attention_mask','target_indices']  #bert时
        self.inputs_cols = ['text_indices','bert_mask','bert_segment','source_mask','sentiment_mask','maskItem_input','maskRel_input']  #ABSA时
        self.criterion = torch.nn.CrossEntropyLoss()


    '''训练入口'''
    def train(self):
        self.best_acc = 0
        self.best_f1 = 0
        self.best_mif1 = 0
        self.best_avf1 = 0
        self.best_pr = 0
        self.best_re = 0
        self.best_f1_favor = 0
        self.best_f1_against = 0
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

        for epoch in range(self.args.epochs):
            # train for one epoch
            print("Begin training....")

            self.train_per_epoch(epoch)
            #验证
            self.val_per_epoch(epoch)
            # self.logger.save_curves(epoch)
            # self.logger.save_check_point(self.model, epoch)

        # torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, "KMELM_model.tar")
        # torch.save(self.model, 'KMELM_model2.pth')

        print("\n"
              "+++++++++++After all epoch, best_Accuracy : {}, precision : {}, recall: {}, f1 : {}, micro_f1 : {}, avg_f1 : {}".format(self.best_acc,
                                                                                                                        self.best_pr,self.best_re,
                                                                                                                        self.best_f1,self.best_mif1,
                                                                                                                                       self.best_avf1))
        if is_MT == True:
            print("au_task_acc :{}, au_task_pr :{}, au_task_re :{} ,au_task_f1 : {}, au_task_micro_f1 : {}, au_task_avg_f1 : {}".format(self.best_au_acc, self.best_au_pr, self.best_au_re,self.best_au_f1, self.best_au_mif1, self.best_au_avf1))

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()

        for i, sample_batched in enumerate(self.train_data_loader):
            # losses = []
            if is_MT == False:
                x, aspect_label, aspect_pred, opinion_label, opinion_pred, sentiment_label, sentiment_pred, tsmtd_label, tsmtd_pred, prd_label, prd_pred = self.step(sample_batched)
                # print(x)
                # compute loss
                aspect_loss = torch.mean(self.criterion(aspect_pred, torch.argmax(aspect_label, dim=1)))
                opinion_loss = torch.mean(self.criterion(opinion_pred, torch.argmax(opinion_label, dim=1)))
                sentiment_loss = torch.mean(self.criterion(sentiment_pred, torch.argmax(sentiment_label, dim=1)))
                tsmtd_loss = torch.mean(self.criterion(tsmtd_pred, tsmtd_label))
                prd_loss = torch.mean(self.criterion(prd_pred, prd_label))

                loss_absa = aspect_loss + opinion_loss + sentiment_loss
                loss_aux = tsmtd_loss + prd_loss
                loss = loss_absa + 1/2 * loss_aux  # α=0.5

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.opt_c.zero_grad()
            # self.opt_g.zero_grad()
            # loss.backward()
            # self.opt_c.step()
            # self.opt_g.step()


            if i % 10 == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))
                #每次都有bachsize个


    def val_per_epoch(self, epoch):
        self.model.eval()
        dev_a_preds, dev_a_labels = [], []
        dev_o_preds, dev_o_labels = [], []
        dev_s_preds, dev_s_labels = [], []
        dev_final_mask = []
        for i, sample_batched in enumerate(self.dev_data_loader):
            x, aspect_label, aspect_pred, opinion_label, opinion_pred, sentiment_label, sentiment_pred, tsmtd_label, tsmtd_prob, prd_label, prd_prob = self.step(sample_batched)
            mask = sample_batched['source_mask']
            mask = mask.to(device)
            dev_a_preds.extend(aspect_pred)
            dev_a_labels.extend(aspect_label)
            dev_o_preds.extend(opinion_pred)
            dev_o_labels.extend(opinion_label)
            dev_s_preds.extend(sentiment_pred)
            dev_s_labels.extend(sentiment_label)
            dev_final_mask.extend(mask)

        dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1 \
            = get_metric(dev_a_labels, dev_a_preds, dev_o_labels, dev_o_preds, dev_s_labels, dev_s_preds,
                         dev_final_mask, 1)

        print(
            "Epoch {} complete! aspect_F1 : {}, opinion_F1 : {}, sentiment_acc : {}, sentiment_F1: {}, ABSA_F1 : {}".format(
                                                                        epoch, dev_aspect_f1, dev_opinion_f1,
                                                                          dev_sentiment_acc, dev_sentiment_f1
                                                                          , dev_ABSA_f1))

    '''进行数据拆包'''
    def step(self, sample_batched):

        x = [sample_batched[col].to(device) for col in self.inputs_cols]

        aspect_label = torch.reshape(sample_batched['aspect'], [-1, 3])
        aspect_label = aspect_label.to(device)
        opinion_label = torch.reshape(sample_batched['opinion'], [-1, 3])
        opinion_label = opinion_label.to(device)
        sentiment_label =torch.reshape(sample_batched['sentiment'], [-1, 3])
        sentiment_label = sentiment_label.to(device)
        tsmtd_label = sample_batched['maskType'].to(device)
        prd_label = sample_batched['rel'].to(device)

        aspect_pred,opinion_pred,sentiment_pred, tsmtd_prob, prd_prob = self.model(x)  #通用

        return x, aspect_label, aspect_pred, opinion_label, opinion_pred, sentiment_label,sentiment_pred, tsmtd_label, tsmtd_prob, prd_label, prd_prob

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
