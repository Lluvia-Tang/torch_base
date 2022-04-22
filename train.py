import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
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
from model.DCRAN.model_bert import DCRAN_BERT
from model.GCAE.myGCAE import GCAE
from model.GCN.mygraph import MYGCN
from model.NES_SGD.NES_SGD import LearnedSharingResNet18
from model.NES_SGD.segnet import SegNet
from model.SCN.StanceCNN import StanceCNN
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
            # lap14_dataset = makeBert_datalodaer.DatesetReader(dataset="covid-19-tweet")
            covid_dataset = makeBert_datalodaer.DatesetReader(dataset="covid-19-tweet")
            # covid_dataset = makeBert_datalodaer.DatesetReader(dataset="semEval2016")

        print("已得到dataset，正在准备dataloader.....")
        self.train_data_loader = BucketIterator(data=covid_dataset.train_data, batch_size=16,
                                                shuffle=True)
        self.test_data_loader = BucketIterator(data=covid_dataset.test_data, batch_size=16,
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
        self.model = BertForClassification(args.bert_path,args.num_classes)  #纯BERT
        # self.model = bertLSTM(args.bert_path,args.num_classes)
        # self.model = Multi_Channel_CNN(len(covid_dataset.embedding_matrix),au_vocab_size_list=[len(get_wordvec("100_tag_embedding_matrix.pkl")),len(get_wordvec("100_position_embedding_matrix.pkl"))],
        #                                embedding_dim=300,hidden_dim=512,n_filters=100,kernel_size=[2,3,3],embed_weights=covid_dataset.embedding_matrix,
        #                                au_weight=[get_wordvec("100_tag_embedding_matrix.pkl"),get_wordvec("100_position_embedding_matrix.pkl")])
        # self.model = CoLSTM(input_size=len(covid_dataset.embedding_matrix),n_filters=16,kernel_size=[2,3,3],embedding_dim=300,hidden_size=512,weights=covid_dataset.embedding_matrix)
        self.model = ABCDM(input_size=len(covid_dataset.embedding_matrix),embedding_dim=300, hidden_size=512, n_filters=32,kernel_size=[2,3,4],weights=covid_dataset.embedding_matrix)
        # self.model = SkepForClassification(args.bert_path, args.num_classes)
        # self.model = T_DAN(input_size=len(covid_dataset.embedding_matrix), embedding_dim=300, hidden_size=512, weights=covid_dataset.embedding_matrix)
        # self.model = StanceCNN(model = args.bert_path, n_filters=16, kernel_size=[2,3,4])

        #多任务模型：
        #ATJSS:
        # self.model = BaseModel(linear_size=len(covid_dataset.embedding_matrix),hidden_size=512,embedding_dim=300, embedding_matrix=covid_dataset.embedding_matrix,dropout=0.5)
        # self.model = SegNet(input_size=len(covid_dataset.embedding_matrix), weights=covid_dataset.embedding_matrix)
        # self.model = LearnedSharingResNet18([3,3],input_size=len(covid_dataset.embedding_matrix),embedding_dim=300,kernel_size=3,num_modules=20,weights=covid_dataset.embedding_matrix)
        #图GNN：
        # self.model = GCN(nfeat=predata.nfeat_dim, nhid=200, nclass=predata.nclass, dropout=0.5)
        # self.model = MYGCN(hidden_size=512,embedding_dim=300, embedding_matrix=covid_dataset.embedding_matrix)

        #ABSA模型
        # self.model = DCRAN_BERT(args.bert_path, hidden_size=512)


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

        self.inputs_cols = ['text_indices','attention_mask','target_indices','tag_indices','position_indices',"graph"]
        # self.inputs_cols = ['text_indices','attention_mask','target_indices']  #bert时
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
        self.tasks = ["Stance", "Sentiment"]

        for epoch in range(self.args.epochs):
            # train for one epoch
            print("Begin training....")

            self.train_per_epoch(epoch)
            #验证
            self.val_per_epoch(epoch)
            # self.logger.save_curves(epoch)
            # self.logger.save_check_point(self.model, epoch)

        # torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, "KMELM_model.tar")
        torch.save(self.model, 'KMELM_model2.pth')

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
                x, label, pred = self.step(sample_batched)
                # print(x)

                # compute loss
                # print(pred)
                # print(label)
                metrics = self.compute_metrics(pred, label, is_train=True)

                # get the item for backward
                loss = metrics['train/l1']
            elif is_MT == True:
                x = [sample_batched[col].cuda() for col in self.inputs_cols]
                stance = sample_batched['stance'].to(device)
                sentiment = sample_batched['sentiment'].to(device)
                st_pred, senti_pred, stance_weight, sentiment_weight = self.model(x)

                # train_pred, logsigma = self.model(x)
                # st_pred = train_pred[0]
                # senti_pred = train_pred[1]
                # stance_weight = logsigma[0]
                # sentiment_weight = logsigma[1]

                #calculate loss
                loss, (stance_loss, senti_loss) = loss_function(
                    stance_predict=st_pred,
                    stance_target=stance,
                    sentiment_predict=senti_pred,
                    sentiment_target=sentiment,
                    # lexicon_vector=train_lexicon,
                    stance_weight=stance_weight,
                    sentiment_weight=sentiment_weight,
                    stance_loss_weight=0.65)
                    # lexicon_loss_weight=config.lexicon_loss_weight,)


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
        if is_MT == False:
            val_acc, val_loss,val_pr, val_re, val_f1, val_mif1, f1_favor, f1_against = evaluate(model=self.model, criterion=nn.CrossEntropyLoss(), dataloader=self.test_data_loader,
                                                      device=device) #task_type=args.task_type target = self.Target
            print(
                "Epoch {} complete! Accuracy : {}, Loss : {}, precision : {}, recall: {}, f1 : {}, micro_f1 : {}, avg_f1 : {}, f1_favor : {}, f1_against : {}".format(
                                                                            epoch, val_acc, val_loss,
                                                                              val_pr, val_re
                                                                              , val_f1, val_mif1,
                                                                              (val_f1 + val_mif1) / 2, f1_favor, f1_against))
        else:

            val_acc, val_loss, val_pr, val_re, val_f1, val_mif1, senti_acc, senti_f1, senti_mif1,senti_re, senti_pr, f1_favor, f1_against = evaluate_MT(model=self.model,
                                                                           dataloader=self.test_data_loader,
                                                                           device=device)


            print(
                "Epoch {} complete! Accuracy : {}, Loss : {}, precision : {}, recall: {}, f1 : {}, micro_f1 : {}, avg_f1 : {}, f1_favor : {}, f1_against : {}\n"
                "senti_acc: {}, senti_pr: {}, senti_re: {}, senti_f1: {}, senti_micro_f1 : {}, senti_avg_f1 : {}".format(epoch, val_acc, val_loss,
                                                                                                  val_pr, val_re
                                                                                                  ,val_f1, val_mif1,
                                                                                                  (val_f1 + val_mif1) / 2, f1_favor, f1_against, senti_acc, senti_pr,senti_re ,senti_f1, senti_mif1,(senti_f1+senti_mif1) / 2))

            # if val_acc >= 0.65 and val_acc <= 0.69:
            #     breakpoint()

        if val_acc > self.best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(self.best_acc, val_acc))
            self.best_acc = val_acc
            self.best_f1 = val_f1
            self.best_mif1 = val_mif1
            self.best_avf1 = (self.best_f1 + self.best_mif1) / 2
            self.best_pr = val_pr
            self.best_re = val_re
            self.best_f1_favor = f1_favor
            self.best_f1_against = f1_against
            if is_MT == True:
                self.best_au_acc = senti_acc
                self.best_au_pr = senti_pr
                self.best_au_re = senti_re
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
        # label = sample_batched['sentiment'].cuda()
        #
        # compute output
        pred = self.model(x)  #通用
        # pred, attention = self.model(x)  #通用
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
