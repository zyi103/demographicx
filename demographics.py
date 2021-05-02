import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, AutoTokenizer
import torch

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
        out = np.argmax(self.model(**inputs)[1].detach().tolist()[0])
        if out == 0:
            return("male")
        elif out == 2:
            return("female")
        else:
            return("unknown")
        
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
        out = np.argmax(self.model(**inputs)[1].detach().tolist()[0])
        if out == 0:
            return("Black")
        elif out == 1:
            return("Hispanic")
        elif out == 2:
            return("White")
        elif out == 3:
            return("Asian")
        



