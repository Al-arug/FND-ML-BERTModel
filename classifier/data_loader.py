import torch
import torch.nn as nn
import torchtext
from torchtext.data import Field, TabularDataset, BucketIterator
import csv
import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import transformers 

class Loader:
    
    def __init__(self):
        
        super().__init__()
       # self.path = path
        self.device = torch.device('cuda:3')

    def load(self):
        
        fake = pd.read_csv('News-data/Data3/Fake.csv',sep=",",names=["title","text","subject","date"])
        true = pd.read_csv('News-data/Data3/True.csv',sep=",",names=["title","text","subject","date"])
        fake_x = fake.text[1:]
        true_x = true.text[1:]
        fake_y = [1 for i in fake_x]
        true_y = [0 for i in true_x]

        self.x = pd.concat([fake_x,true_x])
        self.y= pd.DataFrame(fake_y+true_y,columns=["label"])
        #self.y=self.y[:1000]
        #self.x=self.x[:1000]
        pass
        
    def tokenize(self):
                
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.load()
        
        
        self.input_ids = []
        self.attention_masks = []
       # max_length=max([len(text) for text in x ])

        for text in self.x.tolist() :
            encoded_dict = tokenizer.encode_plus(
                       text ,                      #Article to encode
                        add_special_tokens = True, #Add [CLS] to start, [SEP] to end
                        max_length = 70,
                        truncation=True ,           #Pad & truncate all sentences
                        pad_to_max_length = True ,
                        #padding=True,
                        return_attention_mask = True,   #Create masks
                        return_tensors = 'pt', 
                        # Return pytorch tensors
                   )
    
   
            self.input_ids.append(encoded_dict['input_ids'])
            self.attention_masks.append(encoded_dict['attention_mask'])
        
        pass
        
    def batcher(self):
        self.tokenize()
        
        input_ids = torch.cat(self.input_ids, dim=0) 
        attention_masks = torch.cat(self.attention_masks, dim=0) #mask
        labels = torch.tensor(self.y.label.tolist()) #labels (fake/true)


        labels = torch.tensor(labels) 
        dataset = TensorDataset(input_ids, attention_masks, labels)


        train_size = int(0.7 * len(dataset)) +1
        val_size = int(0.1 * len(dataset)) +1 
        test_size = int(0.2 * len(dataset)) 


        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


        batch_size = 16

        self.train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size)

        self.validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially, since order doesn't matter here
            batch_size = batch_size)

        self.test_dataloader = DataLoader(
            test_dataset, 
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially, since order doesn't matter here
            batch_size = batch_size)
        
        return self.train_dataloader, self.validation_dataloader, self.test_dataloader
        
        
        
        
        
        
        
        
        
        
  
    
    