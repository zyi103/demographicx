import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from scipy.special import softmax
from transformers import BertForSequenceClassification, AutoTokenizer


def get_name_pair(s):
        return(s, ' '.join(str(s)).replace('  ', ' ').replace('  ', ' '))

class gender_classifier:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained("liamliang/demographics_gender")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    def predict(self, name):
        encoded = self.tokenizer.batch_encode_plus([get_name_pair(str(name).lower())], return_attention_mask=True, padding=True, return_tensors='pt')
        dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'], torch.tensor([0])) 
        dataloader = DataLoader(dataset, batch_size=1)
        for batch in dataloader:
            inputs = {'input_ids':  batch[0], 'attention_mask': batch[1], 'labels': batch[2],}
        output = softmax(self.model(**inputs)[1].detach().tolist()[0])
        res = {}
        res['male'] = output[0]
        res['unknown'] = output[1]
        res['female'] = output[2]
        return(res)

        
class race_classifier:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained("liamliang/demographics_race")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    def predict(self, name):
        encoded = self.tokenizer.batch_encode_plus([get_name_pair(str(name).lower())], return_attention_mask=True, padding=True, return_tensors='pt')
        dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'], torch.tensor([0])) 
        dataloader = DataLoader(dataset, batch_size=1)
        for batch in dataloader:
            inputs = {'input_ids':  batch[0], 'attention_mask': batch[1], 'labels': batch[2],}
        output = softmax(self.model(**inputs)[1].detach().tolist()[0])
        res = {}
        res['Black'] = output[0]
        res['Hispanic'] = output[1]
        res['White'] = output[2]
        res['Asian'] = output[3]
        return(res)




