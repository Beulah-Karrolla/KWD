import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import conformer
from conformer import ConformerBlock

class TempSpeechClassifier(nn.Module):
    def __init__(self):
        super.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(32*126*126, 128)
        self.fc2 = nn.Linear(128,1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
class SpeechClassifierModel(nn.Module):
    def __init__(self, num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional, device):
        super(SpeechClassifierModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = 2 if bidirectional else 1
        self.device = device  
        self.projTr = nn.Linear(768, 256)
        self.projSr = nn.Linear(40, 256)  
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=dropout, 
                            bidirectional=bidirectional, batch_first=True)
        
        self.fc = nn.Linear(hidden_size*self.direction, num_classes)

    #def forward(self, x, y):
    def forward(self, logmels, bEmb):
        #import ipdb; ipdb.set_trace()
        #logmels_out = self.logmels_transformer(logmels)
        #logmels_out = self.projSr(logmels_out)
        logmels_out = self.projSr(logmels)

        #bpe_out = self.bpe_transformer(bEmb)
        #bEmb_out = self.projTr(bpe_out)
        bEmb_out = self.projTr(bEmb)

        out_features = torch.cat((logmels_out, bEmb_out), dim=1)
        #out,(hn,cn) = self.lstm(x)
        out,(hn,cn) = self.lstm(out_features)
        out = self.fc(out)
        #out = nn.AvgPool1d(1)(out)
        out = nn.AdaptiveAvgPool1d(1)(out.permute(0,2,1))
        out = torch.sigmoid(out)
        return out

#class ConformerModel(nn.Module):
class ConformerModel(nn.Module):
    def __init__(self, num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional, device):
        super(ConformerModel, self).__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.direction = 2 if bidirectional else 1
        self.device = device
        self.num_blocks = 16
        
        self.logmels_encoder = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.logmels_transformer = nn.TransformerEncoder(self.logmels_encoder, num_layers=num_layers)
        
        #self.bpe_encoder = nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=256, dropout=dropout)
        #self.bpe_transformer = nn.TransformerEncoder(self.bpe_encoder, num_layers=num_layers)
        
        self.projTr = nn.Linear(768, 256)
        self.projSr = nn.Linear(40, 256)

        self.conformer_blocks = nn.ModuleList([ConformerBlock(dim=256, 
                                                              dim_head=64, 
                                                              heads=4, 
                                                              ff_mult=4, 
                                                              conv_expansion_factor=2, 
                                                              conv_kernel_size=31, 
                                                              attn_dropout=0.1, 
                                                              ff_dropout=0.1, 
                                                              conv_dropout=0.1) 
                                                              for _ in range(self.num_blocks)])

        self.fc = nn.Linear(hidden_size*self.direction, num_classes)

    def forward(self, logmels, bEmb):
        #import ipdb; ipdb.set_trace()
        #logmels_out = self.logmels_transformer(logmels)
        #logmels_out = self.projSr(logmels_out)
        import ipdb; ipdb.set_trace()
        logmels_out = self.projSr(logmels)

        #bpe_out = self.bpe_transformer(bEmb)
        #bEmb_out = self.projTr(bpe_out)
        bEmb_out = self.projTr(bEmb)

        out_features = torch.cat((logmels_out, bEmb_out), dim=1)
        for block in self.conformer_blocks:
            out_features = block(out_features)
        import ipdb; ipdb.set_trace()
        out = self.fc(out_features)
        out = nn.AdaptiveAvgPool1d(1)(out.permute(0, 2, 1))
        out = torch.sigmoid(out)
        return out

    '''def forward(self, x):
        import ipdb; ipdb.set_trace()
        out = self.conformer(x)
        out = self.fc(out)
        out = nn.AdaptiveAvgPool1d(1)(out.permute(0,2,1))
        out = torch.sigmoid(out)
        return out'''

   
class SpeechClassifierModelTransformer(nn.Module):
    def __init__(self, num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional, device):
        super(SpeechClassifierModelTransformer, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = 2 if bidirectional else 1
        self.device = device
        #self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size*self.direction, nhead=4, dim_feedforward=256, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dim_feedforward=256, dropout=0.2)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size*self.direction, num_classes)

    def forward(self, x):
        out = self.transformer(x)
        out = self.fc(out)
        #out = nn.AvgPool1d(1)(out)
        out = nn.AdaptiveAvgPool1d(1)(out.permute(0,2,1))
        out = torch.sigmoid(out)
        return out
    
        ''' _, (hn, _) = self.lstm(x)
        hn = hn.transpose(0,1)
        _, (hn2, _) = self.lstm2(hn)
        hn2 = hn2.transpose(0,1)
        _, (hn3, _) = self.lstm3(hn2)
        out = self.fc(hn3)
        out = torch.sigmoid(out[-1])
        return out'''

    # Example usage:
    '''num_classes = 1
    feature_size = 40
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    bidirectional = True

    model = SpeechClassifierModel(num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional)

'''

