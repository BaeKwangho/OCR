import numpy as np
import torch
import fastwer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_cer(pred,target):
    for i,char in enumerate(pred):
        if char=='<s>':
            pred = pred[:i]
            break
    for i,char in enumerate(target):
        if char=='<s>':
            target = target[:i]
            break
            
    new_text = str()
    for i in pred:
        if not '가'<=i<='힣':
            continue
        new_text+=i
    
    return fastwer.score_sent(target, new_text, char_level=True)
   

class ScoreCalc(object):
    def __init__(self):
        self.all_score = 0
        self.num = 0
        
    def add(self, target, preds,length_true):
        batch_size = target.shape[0]
        length = target.shape[1]
        
        pred_max = torch.argmax(preds,2)
        acc = 0
        
        one_hot = torch.zeros(preds.shape).to(device)
        one_hot_pred = torch.zeros(preds.shape).to(device)
        
        for i,_ in enumerate(preds):
            one_hot_pred[i,torch.arange(length),pred_max[i]]=1
            one_hot[i,torch.arange(length),target[i]]=1
        
            temp = torch.sum(one_hot_pred[i,:length_true[i]-1]*one_hot[i,:length_true[i]-1])
            acc += temp
            
        length_sum = sum([i for i in length_true])    
        per = acc/length_sum
        
        self.all_score+=per
        self.num+=1
        
        del one_hot,one_hot_pred,temp, pred_max
        
    def val(self):
        return (self.all_score/self.num)*100
    
class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res