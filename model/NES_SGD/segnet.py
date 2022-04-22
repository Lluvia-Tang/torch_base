# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/25 19:42'
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

class SegNet(nn.Module):
    def __init__(self, input_size, weights = None):
        super(SegNet, self).__init__()
        # initialise network parameters
        # filter = [64, 128, 256, 512, 512]
        filter = [64]
        self.class_nb = 3
        embedding_dim = 300

        # define encoder decoder layers
        self.encoder_block_t = nn.ModuleList([nn.ModuleList([self.conv_layer([1, filter[0], filter[0]], bottle_neck=True)])])
        self.decoder_block_t = nn.ModuleList([nn.ModuleList([self.conv_layer([filter[0], filter[0], filter[0]], bottle_neck=True)])])

        for j in range(2):
            if j < 1:
                self.encoder_block_t.append(nn.ModuleList([self.conv_layer([1, filter[0], filter[0]], bottle_neck=True)]))
                self.decoder_block_t.append(nn.ModuleList([self.conv_layer([filter[0], filter[0], filter[0]], bottle_neck=True)]))
            # for i in range(4):
            #     if i == 0:
            #         self.encoder_block_t[j].append(self.conv_layer([filter[i], filter[i + 1], filter[i + 1]], bottle_neck=True))
            #         self.decoder_block_t[j].append(self.conv_layer([filter[i + 1], filter[i], filter[i]], bottle_neck=True))
            #     else:
            #         self.encoder_block_t[j].append(self.conv_layer([filter[i], filter[i + 1], filter[i + 1]], bottle_neck=False))
            #         self.decoder_block_t[j].append(self.conv_layer([filter[i + 1], filter[i], filter[i]], bottle_neck=False))

        # define cross-stitch units
        self.cs_unit_encoder = nn.Parameter(data=torch.ones(4, 2))
        self.cs_unit_decoder = nn.Parameter(data=torch.ones(5, 2))

        # define task specific layers
        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], bottle_neck=True, pred_layer=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], bottle_neck=True, pred_layer=True)


        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.down_sampling = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        #[F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5]))
        if weights is not None:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, _weight=weights)
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                nn.init.constant(m.weight, 1)

    def conv_layer(self, channel, bottle_neck, pred_layer=False):
        if bottle_neck:
            if not pred_layer:
                conv_block = nn.Sequential(
                    nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channel[1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channel[2]),
                    nn.ReLU(inplace=True),
                )
            else:
                conv_block = nn.Sequential(
                    nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                    nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
                )

        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(channel[2]),
                nn.ReLU(inplace=True),
            )
        return conv_block

    def forward(self, x):
        x = x[0]
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        # embedded = [batchsize, seq_len, embedding_dim]
        embedded = embedded.unsqueeze(1)

        encoder_conv_t, decoder_conv_t, encoder_samp_t, decoder_samp_t, indices_t = ([0] * 2 for _ in range(5))
        for i in range(2):
            encoder_conv_t[i], decoder_conv_t[i], encoder_samp_t[i], decoder_samp_t[i], indices_t[i] = ([0] * 5 for _ in range(5))

        # task branch 1
        # for i in range(5):
        for i in range(1):
            for j in range(2):
                if i == 0:
                    encoder_conv_t[j][i] = self.encoder_block_t[j][i](embedded)
                    encoder_samp_t[j][i], indices_t[j][i] = self.down_sampling(encoder_conv_t[j][i])
                    # encoder_samp_t[j][i], indices_t[j][i] = F.max_pool1d(encoder_conv_t[j][i], encoder_conv_t[j][i].shape[2], return_indices=True).squeeze(2)
                else:
                    encoder_cross_stitch = self.cs_unit_encoder[i - 1][0] * encoder_samp_t[0][i - 1] + \
                                           self.cs_unit_encoder[i - 1][1] * encoder_samp_t[1][i - 1]

                    encoder_conv_t[j][i] = self.encoder_block_t[j][i](encoder_cross_stitch)
                    encoder_samp_t[j][i], indices_t[j][i] = self.down_sampling(encoder_conv_t[j][i])

        # print(encoder_conv_t[0][0].shape,"++++++") 64, 64, 20, 300
        # print(encoder_samp_t[0][0].shape,"++++++++") 64, 64, 20, 150
        # print(indices_t[0][0].shape,"+++++++") 64, 64, 20, 150

        # for i in range(5):
        for i in range(1):
            for j in range(2):
                if i == 0:
                    decoder_cross_stitch = self.cs_unit_decoder[i][0] * encoder_samp_t[0][-1] + \
                                           self.cs_unit_decoder[i][1] * encoder_samp_t[1][-1]

                    decoder_samp_t[j][i] = self.up_sampling(decoder_cross_stitch, indices_t[j][-i - 1])
                    decoder_conv_t[j][i] = self.decoder_block_t[j][-i - 1](decoder_samp_t[j][i])
                else:
                    decoder_cross_stitch = self.cs_unit_decoder[i][0] * decoder_conv_t[0][i - 1] + \
                                           self.cs_unit_decoder[i][1] * decoder_conv_t[1][i - 1]

                    decoder_samp_t[j][i] = self.up_sampling(decoder_cross_stitch, indices_t[j][-i - 1])
                    decoder_conv_t[j][i] = self.decoder_block_t[j][-i - 1](decoder_samp_t[j][i])

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(decoder_conv_t[0][-1]), dim=1)
        t2_pred = self.pred_task2(decoder_conv_t[1][-1])

        return [t1_pred, t2_pred], self.logsigma
