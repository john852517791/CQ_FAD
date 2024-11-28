import numpy as np
import soundfile as sf
import torch,os
from torch import Tensor
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from .RawBoost import process_Rawboost_feature
import lightning as L
from PIL import Image
from torchvision.transforms import ToTensor


___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"

    


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            # _, key, _, _, _ = line.strip().split(" ")
            # file_list.append(key)
            file_list.append(line)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

 

class Dataset_ASVspoof2019(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        y = self.labels[key]
                # img = Image.open(os.path.join(self.base_dir,key+".png"))
                # img_rgb = img.convert("RGB")
                # # img_rgb.save("example_rgb.jpg")
                # img_array = np.array(img_rgb)
                # img_ar = Image.fromarray(img_array)
                # img_ar = self.totensor(img_ar)
        data = np.load(os.path.join(self.base_dir,key+".npy"))
        len = 256
        if data.shape[2]>len:
            stt = np.random.randint(data.shape[2] - len)
            data = data[:,:,stt:stt + len]
        
        img_ar = Tensor(data)
        return img_ar, y

class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
                # img = Image.open(os.path.join(self.base_dir,key+".png"))
                # img_rgb = img.convert("RGB")
                # # img_rgb.save("example_rgb.jpg")
                # img_array = np.array(img_rgb)
                # img_ar = Image.fromarray(img_array)
                # img_ar = self.totensor(img_ar)
        data = np.load(os.path.join(self.base_dir,key+".npy"))
        img_ar = Tensor(data)
        return img_ar, key
        

class Dataset_ASVspoof2019_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # spk, key, _, type, realorspoof = line.strip().split(" ")
        key = self.list_IDs[index].strip().split(" ")
                # img = Image.open(os.path.join(self.base_dir,key[1]+".png"))
                # img_rgb = img.convert("RGB")
                # # img_rgb.save("example_rgb.jpg")
                # img_array = np.array(img_rgb)
                # img_ar = Image.fromarray(img_array)
                # img_ar = self.totensor(img_ar)
        data = np.load(os.path.join(self.base_dir,key[1]+".npy"))
        img_ar = Tensor(data)
        return img_ar, key[1],key[3],key[4]


class asvspoof_dataModule(L.LightningDataModule):
        def __init__(self,args):
                super().__init__()
                self.args = args
                # 标签文件
                self.protocols_path = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/datasets/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/"
                self.train_protocols_file = self.protocols_path + "ASVspoof2019.LA.cm.train.trl.txt"
                self.dev_protocols_file = self.protocols_path + "ASVspoof2019.LA.cm.dev.trl.txt"
                # 数据集音频路径
                self.dataset_base_path="/data8/wangzhiyong/project/fakeAudioDetection/maeFAD/datasets/asvspoof2019/pic512/"
                self.train_set=self.dataset_base_path+"train/"
                self.dev_set=self.dataset_base_path+"dev/"
                # 测试集
                self.eval_protocols_file_19 = self.protocols_path + "ASVspoof2019.LA.cm.eval.trl.txt"
                self.eval_set_19 = self.dataset_base_path+"test/"
 

        def setup(self, stage: str):
            # Assign train/val datasets for use in dataloaders
            if stage == "fit":
                d_label_trn,file_train = genSpoof_list(
                    dir_meta=self.train_protocols_file,
                    is_train=True,
                    is_eval=False
                    )
                
                self.asvspoof19_trn_set = Dataset_ASVspoof2019(
                    list_IDs=file_train,
                    labels=d_label_trn,
                    base_dir=self.train_set 
                    )
   
                _, file_dev = genSpoof_list(
                    dir_meta=self.dev_protocols_file,
                    is_train=False,
                    is_eval=False)
                
                self.asvspoof19_val_set = Dataset_ASVspoof2019_devNeval(
                    list_IDs=file_dev,
                    base_dir=self.dev_set
                    )
   
            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                file_eval = genSpoof_list(
                    dir_meta=self.eval_protocols_file_19,
                    is_train=False,
                    is_eval=True
                    )
                self.asvspoof19_test_set = Dataset_ASVspoof2019_eval(
                    list_IDs=file_eval,
                    base_dir=self.eval_set_19 
                    )
  
                

        def train_dataloader(self):
            return DataLoader(self.asvspoof19_trn_set, batch_size=self.args.batch_size, shuffle=True,drop_last = True,num_workers=4)

        def val_dataloader(self):
            return DataLoader(self.asvspoof19_val_set, batch_size=self.args.batch_size, shuffle=False,drop_last = False,num_workers=4)            

        def test_dataloader(self):                
            datald =  DataLoader(
                self.asvspoof19_test_set,batch_size=self.args.batch_size,
                shuffle=False,num_workers=4
                )
            if "," in self.args.gpuid:
                datald =  DataLoader(
                    self.asvspoof19_test_set,batch_size=self.args.batch_size,
                    shuffle=False,num_workers=4,
                    sampler=DistributedSampler(self.asvspoof19_test_set)
                    )
            return datald
 
      
      
      
      
      
      
      
      
      
      
