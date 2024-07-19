import torch
import torchaudio
import pandas as pd
import speechbrain as sb
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow
import torch.nn as nn
from sklearn.utils import resample 
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel

#device = torch.device('cuda:{:d}'.format(3))
TOK = BertTokenizer.from_pretrained('bert-base-uncased')
#TOK = TOK.to(device)
#BERTMODEL = BertModel.from_pretrained('bert-base-uncased')
#BERTMODEL= BERTMODEL.to(device)
    
class RandomCut(nn.Module):
    """Augmentation technique that randomly cuts start or end of audio"""

    def __init__(self, max_cut=10):
        super(RandomCut, self).__init__()
        self.max_cut = max_cut

    def forward(self, x):
        """Randomly cuts from start or end of batch"""
        side = torch.randint(0, 1, (1,))
        cut = torch.randint(1, self.max_cut, (1,))
        if side == 0:
            return x[:-cut,:,:]
        elif side == 1:
            return x[cut:,:,:] 
        

class SpeechCommandsDataset(torch.utils.data.Dataset):
    def upsample_minority(self, data):
        df_majority = data[(data['Class']==0)]
        df_minority = data[(data['Class']==1)]
        big_len = len(df_majority)
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=big_len, random_state=42)
        data = pd.concat([df_majority, df_minority_upsampled])
        return data

    def downsample_majority(self, data):
        df_majority = data[(data['Class']==0)]
        df_minority = data[(data['Class']==1)]
        small_len = len(df_minority)
        df_majority_downsampled = resample(df_majority, replace=True, n_samples=610, random_state=42)
        data = pd.concat([df_minority, df_majority_downsampled])
        return data

    def __init__(self, data_path, model_type, device, sample_rate=16000):
        self.data = pd.read_csv(data_path)
        if model_type == 'upsample_minority':
            self.data = self.upsample_minority(self.data)
        elif model_type == 'downsample_majority':
            self.data = self.downsample_majority(self.data)
        #import ipdb;ipdb.set_trace()
        self.device = device
        self.sr = sample_rate
        self.filterbank = Filterbank(n_mels=40)
        self.stft = STFT(sample_rate=sample_rate, win_length=25, hop_length=10, n_fft=400)
        #self.deltas = Deltas(input_size=40).to(device)
        #self.context_window = ContextWindow(window_size=151, left_frames=75, right_frames=75)
        #self.input_norm = InputNormalization().to(device)
        #self.projTr = nn.Linear(768, 256)
        #self.projSr = nn.Linear(40, 256)
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.bertmodel = BertModel.from_pretrained('bert-base-uncased')
        #TOK = TOK.to(device)
        #BERTMODEL = BERTMODEL.to(device)
        


    def __len__(self):
        return self.data.shape[0]
    
    def fix_path(self, path):
        #return path.replace('/data/corpora/swb/swb1/data/', '/research/nfs_fosler_1/vishal/audio/swbd/')
        return path.replace('/home/karrolla.1/KWD/', '/home/karrolla.1/KWD1/KWD/')

    def get_segment(self, wavform, sr, start, end):
        #win = (end - start)/2
        start_time = int(max(0,(start-0.05)*sr))
        end_time = int(min((end+0.05)*sr, len(wavform[0])))  
        return wavform[:, start_time:end_time]

    def __getitem__(self, index):
        curr = self.data.iloc[index]
        try:
            wavform, sr = torchaudio.load(self.fix_path(curr['AudioPath']))
            #wavform = self.get_segment(wavform, sr, curr['Start'], curr['End'])
        except:
            print(curr['AudioPath'])
            print("some error in loading audio")
            return None
        wavform = self.get_segment(wavform, sr, curr['Start'], curr['End'])
        wavform = wavform.type('torch.FloatTensor')
        if sr > self.sr:
            wavform = torchaudio.transforms.Resample(sr, self.sr)(wavform)
        features = self.stft(wavform)
        features = spectral_magnitude(features)
        features = self.filterbank(features)
        return features, curr['Class'], curr['AudioPath'], curr['Label']

    def __getitem__old(self, index):
        curr = self.data.iloc[index]
        #import ipdb;ipdb.set_trace()
        try:
            wavform, sr = torchaudio.load(self.fix_path(curr['AudioPath']))
        except:
            print(self.fix_path(curr['AudioPath']))
            print(curr['AudioPath'])
            print("some error in loading audio")
            return None
        #text = curr['Label']
    
    #wavform = self.input_norm(wavform)
        wavform = wavform.type('torch.FloatTensor')
        if sr > self.sr:
            wavform = torchaudio.transforms.Resample(sr, self.sr)(wavform)
        features = self.stft(wavform)
        features = spectral_magnitude(features)
        features = self.filterbank(features)
        #features = self.projSr(features)

        
        #tokens_tensor = torch.tensor([TOK.encode(text, add_special_tokens=True)])
        #tokens_tensor = torch.tensor([TOK.encode(text, add_special_tokens=True)])
        #textBm = BERTMODEL(tokens_tensor)
        #textemB = textBm.last_hidden_state.detach()
        #textB = self.projTr(textemB)
        return features, curr['Class'], curr['AudioPath'], curr['Label']
        

def collate_fn(data):
    fbanks = []
    pholders = []
    labels = []
    texts = []
    texts_ori = []
    for d in data:
        fbank, label, pholder, text = d
        fbank = fbank.squeeze(0) #if fbank.size(1) > 2 else fbank
        #text = text #if text.size(1) > 2 else text
        #import ipdb;ipdb.set_trace()    
        #speechtext = torch.cat((fbank, textB), 1)
        # NOT YET from here on the fbanks represent the input to the model which is the concatenation of the filterbanks and the text embeddings
        fbanks.append(fbank)
        labels.append(label)
        pholders.append(pholder)
        texts_ori.append(text)
    #import ipdb;ipdb.set_trace()    
    #fbanks = torch.nn.utils.rnn.pad_sequence(fbanks, batch_first=True)
    #try:
    fbanks = pack_sequence(fbanks, enforce_sorted=False)
    #fbanks, lenfbanks = pad_packed_sequence(packfbanks, batch_first=True)  # batch, seq_len, feature'''
    labels = torch.tensor(labels)
    #import ipdb;ipdb.set_trace()
    #texts = torch.tensor(texts)
    #text = [x[1] for x in lst if x[0].size(1) > 2] #list of transcripts in the batch
    #textList = torch.tensor([TOK.encode(texts, add_special_tokens=True)])
    textList = TOK(texts_ori).input_ids
    #att_mask = TOK(texts).attention_mask
    textList = [torch.tensor(x) for x in textList]
    texts = pack_sequence(textList, enforce_sorted=False)
    #texts = torch.nn.utils.rnn.pad_sequence(textList, batch_first=True)
    #texts, lentexts = pad_packed_sequence(packtexts, batch_first=True)  # batch, seq_len, feature'''
    '''except:
        print(labels)
        print(pholders)
        print("some size error in texts")
        #import ipdb;ipdb.set_trace()'''
    #texts = texts[0]
    
    return fbanks, labels, pholders, texts, texts_ori


