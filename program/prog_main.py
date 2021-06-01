from utils import Averager, ScoreCalc
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
import PIL.Image as Image
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Program(object):
    def __init__(self, conf, args):
        conf = conf['Program']
        self.lr = conf['learning_rate']
        self.epochs = conf['epochs']
        self.batch_size = conf['batch_size']
        self.args = args
        
        if not os.path.exists(conf['save_path']):
            os.makedirs(conf['save_path'])
            
        self.save_path = conf['save_path']
        
        self.saved_model = conf['saved_model']
        
        
    def train(self, model, dataloader, name, delete):
        if not os.path.exists(os.path.join(self.save_path, name)):
            os.makedirs(os.path.join(self.save_path, name))
        else:
            if not delete:
                raise SyntaxError(f'{os.path.join(self.save_path, name)} is exist.')
        
        ##
        save_folder = os.path.join(self.save_path, name)
        
        classes = model.classes
        model = torch.nn.DataParallel(model).to(device)
        model.train()
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
        loss_avg = Averager()
        calc = ScoreCalc()
        
        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print('Trainable params num : ', sum(params_num))
        
        # optimizer & scheduler
        optimizer = optim.Adam(filtered_parameters, lr=self.lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=self.epochs)
        
        best_loss = 100
        
        
        for epoch in range(self.epochs):
            with tqdm(dataloader, unit="batch") as tepoch:
                for batch, batch_sampler in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch+1} / Batch {batch+1}")

                    img = batch_sampler[0]
                    text = batch_sampler[1][0]
                    length = batch_sampler[1][1]
                    
                    try:
                        if(self.args.choose_model=="ASTER"):
                            preds  = model(img, text[:, :-1], max(length).cpu().numpy())
                        else:
                            preds  = model(img, text[:, :-1], max(length).cpu().numpy())
                    except:
                        print('catched')
                        continue
                    '''
                    
                    if(self.args.choose_model=="ASTER"):
                        preds  = model(img, text[:, :-1], length)
                    else:
                        preds  = model(img, text[:, :-1], max(length).cpu().numpy())
                    '''
                    target = text[:, 1:]
                    cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                    model.zero_grad()
                    cost.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(),5)  # gradient clipping with 5 (Default)
                    
                    optimizer.step()
                    scheduler.step()

                    loss_avg.add(cost)
                    self.batch_size = len(text)
                    pred_max = torch.argmax(F.softmax(preds,dim=2).view(self.batch_size,-1,classes),2)

                    calc.add(target,F.softmax(preds,dim=2).view(self.batch_size,-1,classes),length)
                    #print(dataloader.dataset.converter.decode(target,length),dataloader.dataset.converter.decode(pred_max,length))
                        
                    tepoch.set_postfix(loss=loss_avg.val().item(),acc=calc.val().item())

                    del batch_sampler,cost,pred_max,img,text,length

                    if batch%(50*(20//self.batch_size))==0:
                        log = dict()
                        log['epoch'] = epoch+1
                        log['batch'] = batch+1
                        log['loss'] = loss_avg.val().item()
                        log['acc'] = calc.val().item()

                        with open(os.path.join(save_folder,f'{name}.log'),'a') as f:
                            json.dump(log, f, indent=2)

                        best_loss = loss_avg.val().item()
                        torch.save(model.state_dict(), os.path.join(save_folder,f'{name}.pth'))

    def test(self, model, target_path, dataloader):
        if not os.path.exists(self.saved_model):
            raise FileNotFoundError(f'No such files {self.saved_model}')
         
        test_data_list = os.listdir(target_path)
        for i,target in enumerate(test_data_list):
            image = np.array(Image.open(os.path.join(target_path,target))).astype(np.float32)
            image = np.transpose(image,(2,0,1))[np.newaxis,:3,:,:]
            image = torch.from_numpy(image).to(device)
            test_data_list[i] = image
                
        classes = model.classes
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(self.saved_model, map_location=device))
        
        model.eval()
        with torch.no_grad():
            for image in test_data_list:
                preds = model(image,None,is_train=False)
                pred_max = torch.argmax(F.softmax(preds,dim=2).view(1,-1,classes),2)
                
                length = np.zeros((1,1))
                for i,key in enumerate(torch.squeeze(pred_max)):
                    if key.item()==69:
                        length[0][0] = i+1
                        break
                print(pred_max)
                print(dataloader.converter.decode(pred_max,length))