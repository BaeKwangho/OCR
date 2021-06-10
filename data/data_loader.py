import PIL.Image as Image
import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from jamo import h2j, j2hcj
import time
from utils import AttnLabelConverter


def Load_Loader(conf):
    dataset = ImgDataSet(conf['Basic'])
    num_target = dataset.num_target
    
    return DataLoader(dataset, batch_size=conf['Program']['batch_size'], shuffle=True, collate_fn=dataset.img_collator), num_target
    
def split_loader(dataloader, batch_size):
    test_len = 20000
    d_collate = dataloader.dataset.img_collator
    train_loader, valid_loader = torch.utils.data.random_split(dataloader.dataset, [len(dataloader.dataset)-test_len, test_len])
    train_loader, valid_loader = DataLoader(train_loader, batch_size = batch_size, shuffle=True, collate_fn = d_collate), DataLoader(valid_loader, batch_size = batch_size, shuffle=True, collate_fn = d_collate)
    return train_loader, valid_loader

class ImgDataSet(Dataset):
    def __init__(self,conf):
        if conf['phoneme_type']:
            from data.target_index import phoneme_index
            self.tar2ind,self.ind2tar,num_target = phoneme_index()
        else:
            from data.target_index import character_index
            self.tar2ind,self.ind2tar,num_target = character_index(conf['char_path'])
        
        self.num_target = num_target
        self.phoneme_type = conf['phoneme_type']
        self.converter = AttnLabelConverter(self.tar2ind)
        
        path = conf['train_img_path']
        if not os.path.exists(path):
            raise FileNotFoundError(f'No such file or directory: {path}')
        
        self.img_path = conf['train_img_path']
        self.img_list = [ i for i in os.listdir(self.img_path) if i.split('.')[-1]=='png']
        new_list = []
        if not conf['phoneme_type']:
            for path in self.img_list:
                flag = False
                for char in path.split('_')[0]:
                    if not char in self.tar2ind.keys():
                        flag = True
                        break
                if not flag:
                    new_list.append(path)
            self.img_list = new_list
        else:            
            for path in self.img_list:
                text_jamo = []
                flag=False
                for j in path.split('_')[0]:
                    text_jamo.extend(list(h2j(j)))
                for i in text_jamo:
                    if not i in list(self.tar2ind):
                        flag = True
                        break
                if not flag:
                    new_list.append(path)
            self.img_list = new_list
            
        
    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self,idx):
        filename = self.img_list[idx]
        
        img = Image.open(os.path.join(self.img_path,filename))
        img = np.asarray(img)/255
        img = np.transpose(img.astype(np.float32),(2,0,1))

        text = filename.split('_')[0]
        
        return (img, text)
    
    def img_collator(self,batch):
        batch_size = len(batch)

        img_list = []
        text_list = []
        for img,text in batch:
            img_list.append(img)
            text_list.append(text)

        # image padding
        batch_width = max([i.shape[2] for i in img_list])
        batch_height = max([i.shape[1] for i in img_list])

        img_list = [torch.tensor(k[np.newaxis,:,:,:]) for k in img_list]
        img_batch = torch.zeros(batch_size,3,batch_height,batch_width)

        for i,data in enumerate(img_list):
            img_batch[i,:,:data.shape[-2],:data.shape[-1]] = data

            
        if self.phoneme_type:
            new_text = []
            for i in text_list:
                text_jamo = []
                for j in i:
                    text_jamo.extend(list(h2j(j)))
                new_text.append(text_jamo)
            text_list = new_text
            
        # text padding
        text_max = max([len(i) for i in text_list])
        text_batch = self.converter.encode(text_list,text_max)
        
        
        
        return (img_batch, text_batch)