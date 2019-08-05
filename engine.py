import torch
import scipy.io.wavfile as wav
from python_speech_features import logfbank
import numpy as np
import os
from src.postprocess import Mapper

class LasEngine:
    def __init__(self, model, mapper):
        self.model = model
        self.mapper = mapper
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
   
    @classmethod
    def load(cls, model_path='./las', mapper_path='./', cuda = False): 
        if cuda == False:
            model = torch.load(model_path, map_location = 'cpu')
        else:
            model = torch.load(model_path, model_name)

        mapper = Mapper(mapper_path)
        return cls(model, mapper)

    def wav2logfbank(self, f_path):
        win_size = 0.025
        n_filters = 40
        (rate,sig) = wav.read(f_path)
        fbank_feat = logfbank(sig,rate,winlen=win_size,nfilt=n_filters)
        return fbank_feat

    def transcribe(self, file, ans_len):
        # TODO simplify this
        # TODO hoi a lan ve vu share memory => dead kernel
    

        ft = self.wav2logfbank(file)
        ft = ft.reshape(tuple([1,1] + list(ft.shape)))
        ts = torch.from_numpy(ft)
        ts = ts.to(device = torch.device('cpu'), dtype=torch.float32)

        ts = ts.squeeze(0)
        state_len = torch.sum(torch.sum(ts,dim=-1)!=0,dim=-1)
        state_len = [int(sl) for sl in state_len]

        #ans_len = 330

        ctc_pred, state_len, att_pred, att_maps = self.model(ts, ans_len, state_len=state_len)

        pred = np.argmax(att_pred.cpu().detach(),axis=-1)

        pred = [self.mapper.translate(p,return_string=True) for p in pred ]

        return ' '.join([''.join(tmp.split(' ')) for tmp in pred[0].split('   ')])
