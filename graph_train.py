import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from sklearn import metrics
from torch import nn
from bucket_iterator import BucketIterator
from data import makeBert_datalodaer
from data.face_mask_dataset import make_dataloader
from data.make_datalodaer import DatesetReader

from options import prepare_train_args
from utils.logger import Logger
from metrics import evaluate, evaluate_MT
from options import prepare_train_args
from utils.logger import Logger
from loss import loss_function
from utils.GCN_preData import PrepareData

from model.model_entry import select_model
from model.BERT.base_BERT import BertForClassification
from utils.torch_utils import load_match_dict
from model.BiLSTM.BiLSTM import BiLSTM

from model.kimCNN.KimCNN import KimCNN
# from model.kimCNN.Textcnn import textCNN
from model.TAN.TAN import LSTM_TAN
from model.ATGRU.ATGRU import ATGRU
from model.GCAE.myGCAE import GCAE
from model.BERT.Bert_LSTM import bertLSTM
from model.ATJSS.model import BaseModel
from model.textGCN.model_text_gnn import GCN

device = "cuda:0"
use_Bert = False
is_MT = False
is_graph = True


class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        # train_loader, val_loader, word_vectors, X_lang, Target = make_dataloader(batch_size=100, debug=False, is_train=True)

        # val_loader, word_vectors, X_lang, Target  = make_dataloader(batch_size=100, debug=False, is_train=False)
        # if use_Bert == False:
        #     print("使用glove获得词向量===========")
        #     covid_dataset = DatesetReader(dataset="covid-19-tweet", embed_dim=300)
        # else:
        #     print("使用bert来embedding=========")
        #     covid_dataset = makeBert_datalodaer.DatesetReader(dataset="covid-19-tweet")
        #
        # print("已得到dataset，正在准备dataloader.....")
        # self.train_data_loader = BucketIterator(data=covid_dataset.train_data, batch_size=64,
        #                                         shuffle=True)
        # self.test_data_loader = BucketIterator(data=covid_dataset.test_data, batch_size=64,
        #                                        shuffle=False)


        predata = PrepareData()


        # for i, sample_batched in enumerate(self.train_data_loader):
        # print("+++++max_len = ",sample_batched['max_len'])
        # print("##########",sample_batched['text'])

        # 图GNN：
        self.model = GCN(nfeat=predata.nfeat_dim, nhid=200, nclass=predata.nclass, dropout=0.5)

        self.model = self.model.to(device)

        # 启动 batchNormal 和 dropout
        param_optimizer = list(self.model.named_parameters())
        # 不需要衰减的参数  #衰减：（修正的L2正则化）
        no_decay = ['bias', 'LayerNorm.bias', 'Layerweight']

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

        self.features = predata.features
        self.adj = predata.adj

        self.train_lst = predata.train_lst
        self.val_lst = predata.test_lst
        self.target = predata.target

        self.adj = self.adj.to(device)
        self.features = self.features.to(device)
        self.target = torch.tensor(self.target).long().to(device)
        self.train_lst = torch.tensor(self.train_lst).long().to(device)
        self.val_lst = torch.tensor(self.val_lst).long().to(device)


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

        for epoch in range(self.args.epochs):
            # train for one epoch
            print("Begin training....")
            self.train_per_epoch(epoch)
            # 验证
            self.val_per_epoch(epoch)
            # self.logger.save_curves(epoch)
            # self.logger.save_check_point(self.model, epoch)
        print("\n"
              "+++++++++++After all epoch, best_Accuracy : {}, precision : {}, recall: {}, f1 : {}, micro_f1 : {}, avg_f1 : {}".format(
            self.best_acc,
            self.best_pr, self.best_re,
            self.best_f1, self.best_mif1,
            self.best_avf1))
        if is_MT == True:
            print(
                "au_task_f1 : {}, au_task_micro_f1 : {}, au_task_avg_f1 : {}".format(self.best_au_f1, self.best_au_mif1,
                                                                                     self.best_au_avf1))

    def train_per_epoch(self, epoch=200):
        # switch to train mode
        self.model.train()

        logits = self.model.forward(self.features, self.adj)

        metrics = self.compute_metrics(logits[self.train_lst], self.target[self.train_lst], is_train=True)

        # get the item for backward
        loss = metrics['train/l1']

        # compute gradient and do Adam step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def val_per_epoch(self, epoch):
        self.model.eval()
        if is_MT == False:
            # val_acc, val_loss, val_pr, val_re, val_f1, val_mif1 = evaluate(model=self.model,
            #                                                                criterion=nn.CrossEntropyLoss(),
            #                                                                dataloader=self.test_data_loader,
            #                                                                device=device)  # task_type=args.task_type target = self.Target
            #
            self.criterion = nn.CrossEntropyLoss()
            mean_acc, mean_loss, count = 0, 0, 0
            n_pre, n_labels = None, None
            with torch.no_grad():
                logits = self.model.forward(self.features, self.adj)



                loss = self.criterion(logits[self.val_lst],
                                      self.target[self.val_lst])

            # print(n_pre.cpu())
            # print(n_labels.item())
            # print(n_pre.item())
            pred = torch.max(logits[self.val_lst], 1)[1]
            acc = metrics.accuracy_score(self.target[self.val_lst].cpu(), pred.cpu())
            f1 = metrics.f1_score(self.target[self.val_lst].cpu(), pred.cpu(), labels=[0, 1], average='macro')
            mif1 = metrics.f1_score(self.target[self.val_lst].cpu(), pred.cpu(), labels=[0, 1], average='micro')
            re = metrics.recall_score(self.target[self.val_lst].cpu(), pred.cpu(), labels=[0, 1], average='macro')
            pr = metrics.precision_score(self.target[self.val_lst].cpu(), pred.cpu(), labels=[0, 1], average="macro")

            print(
                "Epoch {} complete! Accuracy : {}, Loss : {}, precision : {}, recall: {}, f1 : {}, micro_f1 : {}, avg_f1 : {}".format(
                    epoch, acc, loss,
                    pr, re
                    , f1, mif1,
                    (f1 + mif1) / 2))
        else:
            # 只计算主任务的acc等
            val_acc, val_loss, val_pr, val_re, val_f1, val_mif1, senti_f1, senti_mif1 = evaluate_MT(model=self.model,
                                                                                                    dataloader=self.test_data_loader,
                                                                                                    device=device)

            print(
                "Epoch {} complete! Accuracy : {}, Loss : {}, precision : {}, recall: {}, f1 : {}, micro_f1 : {}, avg_f1 : {}\n"
                "senti_f1: {}, senti_micro_f1 : {}, senti_avg_f1 : {}".format(epoch, val_acc, val_loss,
                                                                              val_pr, val_re
                                                                              , val_f1, val_mif1,
                                                                              (val_f1 + val_mif1) / 2, senti_f1,
                                                                              senti_mif1, (senti_f1 + senti_mif1) / 2))
        if acc > self.best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(self.best_acc, acc))
            self.best_acc = acc
            self.best_f1 = f1
            self.best_mif1 = mif1
            self.best_avf1 = (self.best_f1 + self.best_mif1) / 2
            self.best_pr = pr
            self.best_re = re
            if is_MT == True:
                self.best_au_f1 = senti_f1
                self.best_au_mif1 = senti_mif1
                self.best_au_avf1 = (self.best_au_f1 + self.best_au_mif1) / 2

    '''进行数据拆包'''

    def step(self, sample_batched):
        # img, label = data
        # # warp input打包数据
        # img = Variable(img).cuda()
        # label = Variable(label).cuda()
        # x, label, lens = data
        x = [sample_batched[col].cuda() for col in self.inputs_cols]
        # x = sample_batched['text_indices'].cuda()
        label = sample_batched['stance'].cuda()

        # compute output
        pred = self.model(x)  # 通用
        # pred = self.model(img)
        # pred = self.model(x)    #带target嵌入
        return x, label, pred

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
