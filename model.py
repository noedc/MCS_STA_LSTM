from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation
import torch.nn.functional as F
import torch.nn.functional as F


class highwayNet(nn.Module):

    ## Initialization
    def __init__(self, args):
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args['use_maneuvers']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']  # 64
        self.decoder_size = args['decoder_size']  # 128
        self.in_length = args['in_length']  # 16
        self.out_length = args['out_length']  # 25
        self.grid_size = args['grid_size']  # 13, 3
        self.soc_conv_depth = args['soc_conv_depth']  # 64
        self.conv_3x1_depth = args['conv_3x1_depth']  # 16
        self.dyn_embedding_size = args['dyn_embedding_size']  # 80
        self.input_embedding_size = args['input_embedding_size']  # 32
        self.num_lat_classes = args['num_lat_classes']  # 3    横轴 左、右、不变
        self.num_lon_classes = args['num_lon_classes']  # 2    纵轴 减速、不变
        self.soc_embedding_size = (((args['grid_size'][0] - 4) + 1) // 2) * self.conv_3x1_depth

        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        self.tanh = nn.Tanh()

        self.pre4att = nn.Sequential(nn.Linear(self.encoder_size, 1),)
        self.pre4att2 = nn.Sequential(nn.Linear(80, 1),)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv3x3 = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
        self.soc_conv5x3 = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, (5, 3), padding=(1, 0))
        self.soc_conv7x3 = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, (7, 3), padding=(2, 0))
        self.conv_3x1 = torch.nn.Conv2d(192, self.conv_3x1_depth, (3, 1))
        self.soc_maxpool = torch.nn.MaxPool2d((2, 1), padding=(1, 0))

        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # Decoder LSTM
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(
                self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes,
                self.decoder_size)
        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size, 5)
        self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    ## Forward Pass
    def forward(self, hist, nbrs, masks, lat_enc, lon_enc):

        ## Forward pass hist:
        lstm_out, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        # print('lstm_out=', lstm_out.shape)  # torch.Size([16, 32, 64])
        lstm_out = lstm_out.permute(1, 0, 2)
        # print('lstm_out=', lstm_out.shape)  # torch.Size([32, 16, 64])
        lstm_weight = self.pre4att(self.tanh(lstm_out))
        # print('lstm_weight=', lstm_weight.shape)  # torch.Size([32, 16, 1])
        new_hidden, soft_attn_weights = self.attention(lstm_weight, lstm_out)
        # print('new_hidden=', new_hidden.shape)  # torch.Size([32, 64])
        new_hidden = self.leaky_relu(self.dyn_emb(new_hidden.view(hist_enc.shape[1], hist_enc.shape[2])))
        # print('new_hidden=', new_hidden.shape)

        new_hidden = new_hidden.unsqueeze(2)
        # print('new_hidden=', new_hidden.shape)  # new_hidden= torch.Size([32, 64, 1])
        ## Forward pass nbrs
        nbrs_out, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        # print('nbrs_out=', nbrs_out.shape)
        # apply attention mechanism to neighbors
        nbrs_out = nbrs_out.permute(1, 0, 2)
        # print('nbrs_out=', nbrs_out.shape)

        nbrs_lstm_weight = self.pre4att(self.tanh(nbrs_out))
        # print('nbrs_lstm_weight=', nbrs_lstm_weight.shape)

        new_nbrs_hidden, soft_nbrs_attn_weights = self.attention(nbrs_lstm_weight, nbrs_out)
        nbrs_enc = new_nbrs_hidden
        # print('nbrs_enc=', nbrs_enc.shape)

        ## Masked scatter
        masks = masks.bool()
        soc_enc = torch.zeros_like(masks).float()  # mask size: (128, 3, 13, 64)
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        # print('soc_enc=', soc_enc.shape)

        # masks_tem = masks.permute(0, 3, 2, 1)

        soc_enc = soc_enc.permute(0, 3, 2, 1)
        # soc_enc = soc_enc.contiguous().view(soc_enc.shape[0], soc_enc.shape[1], -1)
        # print('soc_enc=', soc_enc.shape)

        ## Apply convolutional social pooling:
        # soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        a = self.soc_conv3x3(soc_enc)
        #print('a=', a.shape)
        b = self.soc_conv5x3(soc_enc)
        #print('b=', b.shape)
        c = self.soc_conv7x3(soc_enc)
        #print('c=', c.shape)
        d = self.leaky_relu(torch.cat((a, b, c), dim=1))
        e = self.leaky_relu(self.conv_3x1(d))
        soc_enc = self.soc_maxpool(e)
        # print('soc_enc=', soc_enc.shape)
        soc_enc = soc_enc.contiguous().view(soc_enc.shape[0], soc_enc.shape[1], -1)
        # print('soc_enc=', soc_enc.shape)
        soc_enc = soc_enc.view(-1, self.soc_embedding_size)
        soc_enc = soc_enc.unsqueeze(2)

        # concatenate hidden states:
        new_hs = torch.cat((soc_enc, new_hidden), 2)
        # print(new_hs.shape)  # torch.Size([32, 64, 40])
        new_hs_per = new_hs.permute(0, 2, 1)
        # print(new_hs_per.shape)  # torch.Size([32, 40, 64])

        # second attention
        weight = self.pre4att2(self.tanh(new_hs_per))

        new_hidden_ha, soft_attn_weights_ha = self.attention(weight, new_hs_per)

        ## Concatenate encodings:
        enc = new_hidden_ha
        enc = enc.unsqueeze(2)
        # print('enc=', enc.shape)
        # print(soc_enc.shape)
        # soc_enc = soc_enc.view(-1, self.soc_embedding_size)
        # print(soc_enc.shape)

        ## Apply fc soc pooling
        # soc_enc = soc_enc.contiguous()
        # soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
        # soc_enc = self.leaky_relu(self.soc_fc(soc_enc))

        ## Concatenate encodings:
        enc = torch.cat((enc, new_hidden), 1)
        # print('enc=', enc.shape)
        enc = enc.squeeze(2)
        # print('enc=', enc.shape)

        if self.use_maneuvers:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## Concatenate 连接 maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)  # 1表示在第二个维度上拼接
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc)
            return fut_pred

    def decode(self, enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred

    def attention(self, lstm_out_weight, lstm_out):
        alpha = F.softmax(lstm_out_weight, 1)

        lstm_out = lstm_out.permute(0, 2, 1)

        new_hidden_state = torch.bmm(lstm_out, alpha).squeeze(2)
        new_hidden_state = F.relu(new_hidden_state)

        return new_hidden_state, alpha

