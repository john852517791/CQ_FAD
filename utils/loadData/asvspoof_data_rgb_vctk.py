import numpy as np
import soundfile as sf
import torch,os
from torch import Tensor
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from .RawBoost import process_Rawboost_feature
import lightning as L
from PIL import Image
from torchvision.transforms import ToTensor
import torchaudio.transforms as T
import matplotlib.pyplot as plt

 
def pad_random(x: np.ndarray, max_len: int = 65600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

class Dataset_ASVspoof2019(Dataset):
    def __init__(self, list_IDs, args):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.args = args
        self.totensor = ToTensor()
        self.spec = T.Spectrogram(n_fft=512)
        self.cmap = plt.get_cmap('hot')

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        X, fs  = sf.read(self.list_IDs[index])
        if self.args.usingDA:
            X=process_Rawboost_feature(X,fs,self.args,self.args.algo)
        X_pad = pad_random(X)
        # to 3channel
        specone = self.spec(Tensor(X_pad))
        rgb_image = self.cmap(specone)[:256,:256,:3]  # 去掉 alpha 通道
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1)
        rgb_tensor = rgb_tensor.numpy()
        x_inp = Tensor(rgb_tensor)
        return x_inp 
   


class asvspoof_dataModule(L.LightningDataModule):
        def __init__(self,args):
                super().__init__()
                self.args = args
                # 数据集音频路径
                self.vctk_wavs = []
                # 打开文本文件并逐行读取
                with open('/data8/wangzhiyong/project/fakeAudioDetection/rgbFAD/vctk.txt', 'r') as file:
                    # 逐行读取文件内容，并添加到数组中
                    for line in file.readlines():
                        # 删除每行末尾的换行符，并添加到数组中
                        self.vctk_wavs.append(line.strip())
                # 数据集音频路径
                self.vctk_eval = []
                # 打开文本文件并逐行读取
                with open('/data8/wangzhiyong/project/fakeAudioDetection/rgbFAD/vctk_eval.txt', 'r') as file:
                    # 逐行读取文件内容，并添加到数组中
                    for line in file.readlines():
                        # 删除每行末尾的换行符，并添加到数组中
                        self.vctk_eval.append(line.strip())

        def setup(self, stage: str):
            # Assign train/val datasets for use in dataloaders
            if stage == "fit":
                self.asvspoof19_trn_set = Dataset_ASVspoof2019(
                     self.vctk_wavs,
                     self.args
                 )
                self.asvspoof19_val_set = Dataset_ASVspoof2019(
                     self.vctk_eval,
                     self.args
                 )

        def train_dataloader(self):
            return DataLoader(self.asvspoof19_trn_set, batch_size=self.args.batch_size, shuffle=True,drop_last = True,num_workers=4)

        def val_dataloader(self):
            return DataLoader(self.asvspoof19_val_set, batch_size=5, shuffle=False,drop_last = False,num_workers=4)            

     
 
      
      
      
      
      
      
      
      
      
      
