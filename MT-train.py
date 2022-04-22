import math

import numpy
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
from tqdm import tqdm

from model.NES_SGD.NES_SGD import LearnedSharingResNet18
from options import prepare_train_args
from utils.logger import Logger
from metrics import evaluate, evaluate_MT
from options import prepare_train_args
from utils.logger import Logger
from loss import loss_function,Bandit_loss_function
from utils.GCN_preData import PrepareData


from model.model_entry import select_model
from model.BERT.base_BERT import BertForClassification
from utils.torch_utils import load_match_dict
from model.BiLSTM.BiLSTM import BiLSTM
from model.kimCNN import text_cnn
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
is_MT = True
is_graph = False

class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        if use_Bert == False:
            print("使用glove获得词向量===========")
            # covid_dataset = DatesetReader(dataset="covid-19-tweet", embed_dim=300)
            covid_dataset = DatesetReader(dataset="semEval2016", embed_dim=300)
        else:
            print("使用bert来embedding=========")
            covid_dataset = makeBert_datalodaer.DatesetReader(dataset="covid-19-tweet")

        print("已得到dataset，正在准备dataloader.....")
        self.train_data_loader = BucketIterator(data=covid_dataset.train_data, batch_size=32,
                                                shuffle=True)
        self.test_data_loader = BucketIterator(data=covid_dataset.test_data, batch_size=32,
                                               shuffle=False)

        if is_graph:
            predata = PrepareData()
            self.feature = predata.features
            self.adj = predata.adj



        #多任务模型：
        #ATJSS:
        # self.model = BaseModel(linear_size=len(covid_dataset.embedding_matrix),hidden_size=512,embedding_dim=300, embedding_matrix=covid_dataset.embedding_matrix,dropout=0.5)

        self.model = dict({})
        self.model['rep'] = text_cnn.TextR(input_size=len(covid_dataset.embedding_matrix), embedding_dim=300, n_filters=16, kernel_size=[2,3,4], weights=covid_dataset.embedding_matrix, net_dropout=0.5)
        for t in ['Stance', 'Sentiment']:
            self.model[t] = text_cnn.TextO(input_size=len(covid_dataset.embedding_matrix), embedding_dim=300, n_filters=16, kernel_size=[2,3,4], weights=covid_dataset.embedding_matrix, net_dropout=0.5)
            # self.model[t].to(device)
        # self.model = self.model.to(device)

        params = {'all': []}
        for m_id, m in enumerate(self.model):
            self.model[m].to(device)
            if (m == 'rep') :
                params[m] = self.model[m].parameters()
            else:
                params['all'] += self.model[m].parameters()


        self.opts = []
        self.schedulers = []
        for m in ['all', 'rep', 'dis']:
            if m in params:
                opt = torch.optim.Adam(params[m], lr=self.args.lr, weight_decay=self.args.weight_decay,betas=(self.args.momentum, self.args.beta))
                self.opts.append(opt)

        self.opt_c, self.opt_g = self.opts
        '''多个任务训练一个模型时'''
        # self.model = LearnedSharingResNet18([3, 3], input_size=len(covid_dataset.embedding_matrix), embedding_dim=300,
        #                                     kernel_size=3, num_modules=20, weights=covid_dataset.embedding_matrix)
        #
        # self.model = self.model.to(device)
        #
        # # 启动 batchNormal 和 dropout
        # param_optimizer = list(self.model.named_parameters())
        # # 不需要衰减的参数  #衰减：（修正的L2正则化）
        # # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'LayerNorm.weight']
        #
        # # 指定哪些权重更新，哪些权重不更新
        # optimizmer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        #     # 遍历所有参数，如果参数名字里有no_decay的元素则取出元素
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #     # 遍历所有参数，如果参数名字里没有no_decay的元素则取出元素
        # ]
        #
        # # self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
        # self.optimizer = torch.optim.Adam(optimizmer_grouped_parameters, self.args.lr,
        #                                   betas=(self.args.momentum, self.args.beta),
        #                                   weight_decay=self.args.weight_decay)

        self.inputs_cols = ['text_indices','attention_mask','target_indices']
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
        self.p_k = [1/2,1/2]
        self.rho = 1.2
        self.eta_p = 0.5
        self.w = [0.1201, 0.2933]
        self.tasks = ["Stance", "Sentiment"]

        for epoch in range(self.args.epochs):
            # train for one epoch
            print("Begin training....")

            # Bandit
            # function for computing lambda


            # self.Tchebycheff_train_per_epoch(epoch)
            self.bandit_train_per_epoch(epoch)
            # self.CAGRAD_train_per_epoch(epoch)
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
            print("au_task_f1 : {}, au_task_micro_f1 : {}, au_task_avg_f1 : {}".format(self.best_au_f1, self.best_au_mif1, self.best_au_avf1))


    def bandit_train_per_epoch(self, epoch):
        # switch to train mode
        for m in self.model:
            self.model[m].train()

        # function for computing lambda
        def compute_l(x, q, rho):
            kk = 1 / (x + 1)
            q_kk = [math.pow(i, kk) for i in q]
            t1 = sum(q_kk)
            t2 = sum([math.log(q[i]) * q_kk[i] for i in range(len(q))]) / (x + 1)
            return math.log(2) - rho - math.log(t1) + t2 / t1

        # Algorithm 2 in paper
        def find_lambda(e, beta, upper, jump):
            if compute_l(0, q_k, self.rho) <= 0:
                return 0
            left = 0
            right = beta
            flag = 0
            while compute_l(right, q_k, self.rho) > 0:
                flag += 1
                left = right
                right = right + beta
                if right > upper:
                    return upper
                if flag > jump:
                    break
            x = (left + right) / 2
            ans = compute_l(x, q_k, self.rho)
            flag = 0
            while abs(ans) > e:
                flag += 1
                if ans > 0:
                    left = x
                else:
                    right = x
                x = (left + right) / 2
                ans = compute_l(x, q_k, self.rho)
                if flag > jump:  # if lambda is too large, skip out the loop
                    return upper
            return x

        train_losses = {t: [] for t in self.tasks}

        for i, sample_batched in enumerate(self.train_data_loader):
            losses = []
            loss = None
            for i, t in enumerate(self.tasks):
                x = [sample_batched[col].cuda() for col in self.inputs_cols]
                stance = sample_batched['stance'].to(device)
                sentiment = sample_batched['sentiment'].to(device)
                y = [stance, sentiment]
                y_ = self.model[t](self.model['rep'](x))
                loss_t = self.criterion(y_, y[i])
                losses.append(loss_t.item())
                if i > 0:
                    loss += loss_t * self.p_k[i]
                else:
                    loss = loss_t * self.p_k[i]

            loss = loss/len(self.tasks)
            self.opt_g.zero_grad()
            self.opt_c.zero_grad()
            loss.backward()
            self.opt_g.step()
            self.opt_c.step()

            q_k = [self.p_k[i] * math.exp(self.eta_p * losses[i]) for i in range(len(self.tasks))]
            lam = find_lambda(1e-15, 10, 2e5, 1e5)
            q_lam = [math.pow(i, 1 / (lam + 1)) for i in q_k]
            q_sum = sum(q_lam)
            self.p_k = [i / q_sum for i in q_lam]


            if i % 10 == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))
                #每次都有bachsize个


    def Tchebycheff_train_per_epoch(self, epoch):
        # switch to train mode
        for m in self.model:
            self.model[m].train()

        train_losses = {t: [] for t in self.tasks}

        for i, sample_batched in enumerate(self.train_data_loader):
            with torch.no_grad():
                # x, y, t_label = batch.text.to(device), batch.label.to(device), batch.task.to(device)
                x = [sample_batched[col].cuda() for col in self.inputs_cols]
                # rep = self.model['rep'](x)
                stance = sample_batched['stance'].to(device)
                sentiment = sample_batched['sentiment'].to(device)
                y = [stance, sentiment]
                rep = self.model['rep'](x)
            self.opt_c.zero_grad()
            for i, t in enumerate(self.tasks):
                xt = rep
                yt = y[i]
                yt_ = self.model[t](xt)
                loss = self.criterion(yt_, yt) * self.w[i]
                train_losses[t].append(loss.item())
                loss.backward()
            self.opt_c.step()


        w = {t: sum(train_losses[t]) / len(train_losses[t]) for t in self.tasks}
        max_t = max(w, key=w.get)
        if max_t == "Stance":
            max_T = 0
        else:
            max_T = 1

        for i, sample_batched in enumerate(self.train_data_loader):
            x = [sample_batched[col].cuda() for col in self.inputs_cols]
            stance = sample_batched['stance'].to(device)
            sentiment = sample_batched['sentiment'].to(device)
            y = [stance, sentiment]
            y_ = self.model[max_t](self.model['rep'](x))

            loss = self.criterion(y_, y[max_T])
            self.opt_g.zero_grad()
            self.opt_c.zero_grad()
            loss.backward()
            self.opt_g.step()
            self.opt_c.step()


            if i % 10 == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))
                #每次都有bachsize个


    def val_per_epoch(self, epoch):
        with torch.no_grad():
            for m in self.model:
                self.model[m].eval()
            accs = []
            losses_var = []
            acc = f1 = mif1 = re = pr = 0
        # self.model.eval()

        if is_MT == True:
            # st_n_pre, st_n_labels = None, None
            # senti_n_pre, senti_n_labels = None, None
            n_pre = [None,None]
            n_labels = [None, None]
            #只计算主任务的acc等
            for ind, t in enumerate(self.tasks):
                test_loss = n_acc = n_all = 0
                for i, sample_batched in enumerate(self.test_data_loader):
                    x = [sample_batched[col].cuda() for col in self.inputs_cols]
                    stance = sample_batched['stance'].to(device)
                    sentiment = sample_batched['sentiment'].to(device)
                    y_ = self.model[t](self.model['rep'](x))
                    y = [stance, sentiment]

                    loss = self.criterion(y_, y[ind])
                    test_loss += loss.item()
                    n_acc += y_.argmax(1).eq(y[ind]).sum()
                    n_all += y[ind].shape[0]

                    if (n_pre[ind] == None):
                        n_pre[ind] = torch.argmax(y_, 1)
                        n_labels[ind] = y[ind]
                        # senti_n_pre = torch.argmax(senti_pred, 1)
                        # senti_n_labels = sentiment
                    else:
                        n_pre[ind] = torch.argmax(y_, 1)
                        n_labels[ind] = y[ind]
                        n_pre[ind] = torch.cat((n_pre[ind], torch.argmax(y_, 1)), dim=0)
                        n_labels[ind] = torch.cat((n_labels[ind], y[ind]), dim=0)
                        # senti_n_pre = torch.cat((senti_n_pre, torch.argmax(senti_pred, 1)), dim=0)
                        # senti_n_labels = torch.cat((senti_n_labels, sentiment), dim=0)



                acc = n_acc / float(n_all)
                accs.append(acc)
                losses_var.append(test_loss)
                print('loss/test/{t} epoch+++++++++', test_loss, epoch)
                print('acc/test/{t} epoch+++++++++', acc, epoch)

                acc = metrics.accuracy_score(n_labels[ind].cpu(), n_pre[ind].cpu())
                f1 = metrics.f1_score(n_labels[ind].cpu(), n_pre[ind].cpu(), labels=[0, 1], average='macro')
                f1_favor = metrics.f1_score(n_labels[ind].cpu(), n_pre[ind].cpu(), labels=[1], average='macro')
                f1_against = metrics.f1_score(n_labels[ind].cpu(), n_pre[ind].cpu(), labels=[0], average='macro')
                mif1 = metrics.f1_score(n_labels[ind].cpu(), n_pre[ind].cpu(), labels=[0, 1], average='micro')
                re = metrics.recall_score(n_labels[ind].cpu(), n_pre[ind].cpu(), labels=[0, 1], average='macro')
                pr = metrics.precision_score(n_labels[ind].cpu(), n_pre[ind].cpu(), labels=[0, 1], average="macro")


                # senti_f1 = metrics.f1_score(senti_n_labels.cpu(), senti_n_pre.cpu(), labels=[0, 1], average='macro')
                # senti_mif1 = metrics.f1_score(senti_n_labels.cpu(), senti_n_pre.cpu(), labels=[0, 1], average='micro')
                print(
                    "Epoch {} complete!For task:{}, Accuracy : {}, precision : {}, recall: {}, f1 : {}, micro_f1 : {}, avg_f1 : {}, f1_favor : {}, f1_against : {}\n".format(epoch,t, acc,
                                                                                  pr, re
                                                                                  , f1, mif1,
                                                                                  (f1 + mif1) / 2, f1_favor, f1_against))
                # if acc >= 0.64 and acc <= 0.66:
                #     breakpoint()

            avg_acc = sum(accs) / len(accs)
            avg_var = numpy.var(losses_var)
            print('avg_acc', avg_acc,"epoch: ", epoch)
            print('avg_var', avg_var,"epoch: ", epoch)


            # if val_acc >= 0.65 and val_acc <= 0.69:
            #     breakpoint()

        if acc > self.best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(self.best_acc, acc))
            self.best_acc = acc
            self.best_f1 = f1
            self.best_mif1 = mif1
            self.best_avf1 = (self.best_f1 + self.best_mif1) / 2
            self.best_pr = pr
            self.best_re = re



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
        pred = self.model(x)  #通用
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
