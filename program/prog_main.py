from utils import Averager, ScoreCalc, get_cer
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
import PIL.Image as Image
import json
from time import time
import unicodedata

class Program(object):
    def __init__(self, conf, args):
        if not conf['Basic']['use_gpu']:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        conf = conf['Program']
        self.lr = conf['learning_rate']
        self.epochs = conf['epochs']
        self.batch_size = conf['batch_size']
        self.args = args
        
        if not os.path.exists(conf['save_path']):
            os.makedirs(conf['save_path'])
            
        self.save_path = conf['save_path']
        
        
    def train(self, model, dataloader, train_loader, valid_loader):
        if not os.path.exists(os.path.join(self.save_path, self.args.name)):
            os.makedirs(os.path.join(self.save_path, self.args.name))
        else:
            if not self.args.delete:
                raise SyntaxError(f'{os.path.join(self.save_path, self.args.name)} is exist.')
        
        save_folder = os.path.join(self.save_path, self.args.name)
        
        classes = model.classes
        model = torch.nn.DataParallel(model).to(self.device)
        model.to(self.device)
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(self.device)

        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print('Trainable params num : ', sum(params_num))
        
        # optimizer & scheduler
        optimizer = optim.Adam(filtered_parameters, lr=self.lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=self.epochs)
        
        best_acc = 0
        taken_time= time()
        for epoch in range(self.epochs):
            t_loss_avg = Averager()
            v_loss_avg = Averager()
            t_calc = ScoreCalc()
            v_calc = ScoreCalc()        
            model.train()
            
            word_target = None
            word_preds = None

            with tqdm(train_loader, unit="batch") as tepoch:
                for batch, batch_sampler in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch+1} / Batch {batch+1}")

                    img = batch_sampler[0].to(self.device)
                    text = batch_sampler[1][0].to(self.device)
                    length = batch_sampler[1][1]
                    
                    if(self.args.choose_model=="ASTER"):
                        preds  = model(img, text[:, :-1], max(length).cpu().numpy())
                    else:
                        preds  = model(img, text[:, :-1], max(length).cpu().numpy())

                    target = text[:, 1:]
                    t_cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                    model.zero_grad()
                    t_cost.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(),5)  # gradient clipping with 5 (Default)
                    
                    optimizer.step()
                    scheduler.step()

                    t_loss_avg.add(t_cost)
                    self.batch_size = len(text)
                    pred_max = torch.argmax(F.softmax(preds,dim=2).view(self.batch_size,-1,classes),2)

                    t_calc.add(target,F.softmax(preds,dim=2).view(self.batch_size,-1,classes),length)
                    #print(dataloader.dataset.converter.decode(target,length),dataloader.dataset.converter.decode(pred_max,length))
                    if batch%(300)==0:
                        word_target = dataloader.dataset.converter.decode(target,length)[0]
                        word_preds = dataloader.dataset.converter.decode(pred_max,length)[0]
                    tepoch.set_postfix(loss=t_loss_avg.val().item(),acc=t_calc.val().item(),\
                                          preds=word_preds,target=word_target)
                    
                    del batch_sampler,pred_max,img,text,length

            model.eval()
            with tqdm(valid_loader, unit="batch") as vepoch:
                for batch, batch_sampler in enumerate(vepoch):
                    vepoch.set_description(f"Epoch {epoch+1} / Batch {batch+1}")
                    with torch.no_grad():
                        img = batch_sampler[0].to(self.device)
                        text = batch_sampler[1][0].to(self.device)
                        length = batch_sampler[1][1].to(self.device)

                        preds  = model(img, text[:, :-1], max(length).cpu().numpy())
                        target = text[:, 1:]
                        v_cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                        torch.nn.utils.clip_grad_norm_(model.parameters(),5)  # gradient clipping with 5 (Default)

                        v_loss_avg.add(v_cost)
                        batch_size = len(text)
                        pred_max = torch.argmax(F.softmax(preds,dim=2).view(batch_size,-1,classes),2)

                        v_calc.add(target,F.softmax(preds,dim=2).view(batch_size,-1,classes),length)

                        vepoch.set_postfix(loss=v_loss_avg.val().item(),acc=v_calc.val().item())
                        del batch_sampler,v_cost,pred_max,img,text,length

            if not os.path.exists(os.path.join(save_folder,self.args.name)):
                os.makedirs(os.path.join(save_folder,self.args.name))
            #save_plt(xs,os.path.join(save_folder,name),0,epoch)
            log = dict()
            log['epoch'] = epoch+1
            log['t_loss'] = t_loss_avg.val().item()
            log['t_acc'] = t_calc.val().item()

            log['v_loss'] = v_loss_avg.val().item()
            log['v_acc'] = v_calc.val().item()
            log['time'] = time() - taken_time
            with open(os.path.join(save_folder,f'{self.args.name}.log'),'a') as f:
                json.dump(log, f, indent=2)

            best_loss = t_loss_avg.val().item()
            if best_acc < v_calc.val().item():
                best_acc = v_calc.val().item()
                torch.save(model.state_dict(), os.path.join(save_folder,f'{self.args.name}.pth'))
            
            
            
    def test(self, model, target_path, dataloader):
        save_folder = os.path.join(self.save_path, self.args.name)
        
        if not os.path.exists(save_folder):
            raise FileNotFoundError(f'No such folders {save_folder}')
        
        classes = model.classes
        model = torch.nn.DataParallel(model).to(self.device)
        model.load_state_dict(torch.load(os.path.join(save_folder,self.args.name+'.pth'), map_location=self.device))
        
        loss_avg = Averager()
        calc = ScoreCalc()
        cer_avg = Averager()
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(self.device)

        model.eval()
        pred_num = 10
        pred_result = []
        with tqdm(dataloader, unit="batch") as vepoch:
            for batch, batch_sampler in enumerate(vepoch):
                vepoch.set_description(f"Test Session / Batch {batch+1}")
                with torch.no_grad():
                    img = batch_sampler[0].to(self.device)
                    text = batch_sampler[1][0].to(self.device)
                    length = batch_sampler[1][1].to(self.device)

                    preds = model(img, text[:, :-1], max(length).cpu().numpy())
                    target = text[:, 1:]
                    v_cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                    torch.nn.utils.clip_grad_norm_(model.parameters(),5)  # gradient clipping with 5 (Default)
                    loss_avg.add(v_cost)
                    batch_size = len(text)
                    pred_max = torch.argmax(F.softmax(preds,dim=2).view(batch_size,-1,classes),2)

                    calc.add(target,F.softmax(preds,dim=2).view(batch_size,-1,classes),length)

                    word_target = dataloader.dataset.converter.decode(target,length)
                    word_preds = dataloader.dataset.converter.decode(pred_max,length)

                    cer_avg.add(torch.from_numpy(np.array(get_cer(word_preds,word_target))))
                    vepoch.set_postfix(loss=loss_avg.val().item(),acc=calc.val().item(),cer=cer_avg.val().item())

                    if batch % (len(vepoch)//10)==0:
                        pred = unicodedata.normalize('NFC',word_preds[0])
                        target = unicodedata.normalize('NFC',word_target[0])
                        pred_result.append(dict(target=target,pred=pred)) 

                    del batch_sampler,v_cost,pred_max,img,text,length
        
        #save_plt(xs,os.path.join(save_folder,name),0,epoch)
        log = dict()
        log['loss'] = loss_avg.val().item()
        log['acc'] = calc.val().item()
        log['cer'] = cer_avg.val().item()
        log['preds'] = pred_result
        
        with open(os.path.join(save_folder,f'{self.args.name}_test.log'),'w') as f:
            json.dump(log, f, indent=2)
            