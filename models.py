import glob
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from natsort import natsorted

import ci_sdr
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class L3DAS_Task1_A_Dataset(Dataset):
    def __init__(self, ddir: str):
        super().__init__()
        self.ddir = ddir
        self.dAdir=natsorted(glob.glob(self.ddir+'/*/*/data/'  +'*_A.wav')) #Aは生データ Bは加工データ
        self.dLdir=natsorted(glob.glob(self.ddir+'/*/*/labels/'+'*.wav'))

    def __getitem__(self, index: int ):
        ditem , sr1 = torchaudio.load(self.dAdir[index])
        litem , sr2 = torchaudio.load(self.dLdir[index])
        assert ditem.size(1) == litem.size(1)
        assert sr1 == sr2
        return ditem,litem

    def __len__(self) -> int:
        return min([len(self.dAdir),len(self.dLdir)])

class MVDRBeamformer(pl.LightningModule):
    def __init__(self,
                  n_fft=1024,
                  kernel_size=3,
                  num_feats=16,
                  num_hidden=64,
                  num_layers=2,
                  num_stacks=4,
                  msk_activate="sigmoid",
                ):
        super().__init__()
        input_dim=(n_fft)//2 +1

        self.maskgeneretor = torchaudio.models.conv_tasnet.MaskGenerator(
            input_dim=input_dim,
            num_sources=2,
            kernel_size=kernel_size,
            num_feats=num_feats,
            num_hidden=num_hidden,
            num_layers=num_layers,
            num_stacks=num_stacks,
            msk_activate=msk_activate
        )
        self.mvdr = torchaudio.transforms.MVDR(
            ref_channel=0,
            solution="stv_evd",
            multi_mask=True
        )
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=n_fft//4,
            power=None,
        )
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=n_fft//4,
        )
    
    # 予測/推論アクションを定義
    def forward(self, x):
        spec = self.stft(x) #[c,t]->[c,f,t]
        mask = self.maskgeneretor(spec.abs()) #[c,f,t]->[b*c,f,t]
        stft_est = self.mvdr(spec, mask[:,0], mask[:,1])
        est = self.istft(stft_est, length=x.shape[-1])
        return est
    
    # train ループを定義
    def training_step(self, batch, batch_idx):  #(ditems,litems)
        x, y = batch  #[b,c,t]
        spec = self.stft(x) #[b,c,t]->[b,c,f,t]
        mask = self.maskgeneretor(spec.abs().view(-1,spec.size(2),spec.size(3))) #[b*c,f,t]->[b*c,f,t]
        stft_est = self.mvdr(spec.view(-1,spec.size(2),spec.size(3)), mask[:,0], mask[:,1]) # ->[b*1,f,t]
        stft_est = stft_est.view(-1,1,spec.size(2),spec.size(3))  #[b*1,f,t]->[b,c(=1),f,t]
        est = self.istft(stft_est, length=x.shape[-1])
        loss = ci_sdr.pt.ci_sdr_loss(est.to(torch.float32), y.to(torch.float32))
        loss = torch.mean(loss)
        self.log("train_loss", loss, logger=True)
        return loss
    
    # validation ループを定義
    def validation_step(self, batch, batch_idx):  #(ditems,litems)
        x, y = batch  #[b,c,t]
        spec = self.stft(x) #[b,c,t]->[b,c,f,t]
        mask = self.maskgeneretor(spec.abs().view(-1,spec.size(2),spec.size(3))) #[b*c,f,t]->[b*c,f,t]
        stft_est = self.mvdr(spec.view(-1,spec.size(2),spec.size(3)), mask[:,0], mask[:,1]) # ->[b*1,f,t]
        stft_est = stft_est.view(-1,1,spec.size(2),spec.size(3))  #[b*1,f,t]->[b,c(=1),f,t]
        est = self.istft(stft_est, length=x.shape[-1])
        loss = ci_sdr.pt.ci_sdr_loss(est.to(torch.float32), y.to(torch.float32))
        loss = torch.mean(loss)
        self.log("val_loss", loss, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def generate_batch(batch):  #[b, getitem]
    lengths = [entry[0].size(1) for entry in batch]
    maxlen  = max(lengths)
    ditems = torch.cat([F.pad(entry[0],pad=(0, maxlen-entry[0].size(1)),mode='constant',value=0)[None] for entry in batch])
    litems = torch.cat([F.pad(entry[1],pad=(0, maxlen-entry[0].size(1)),mode='constant',value=0)[None] for entry in batch])
    assert ditems.size(2)==litems.size(2)
    return ditems,litems