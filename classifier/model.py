import torch
import torch.nn as nn
import torchtext
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn import metrics

class Model:
    def __init__(self,device):
        
        super().__init__()
        self.device=device
        self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab
                 num_labels = 2, #binary classification  
                 output_attentions = False, 
                 output_hidden_states = False)
        
        self.optimizer = AdamW(self.model.parameters(),
                  lr = 0.00002,
                  eps = 1e-8) # default is 1e-8
       
        self.model= self.model.to(self.device)
        
        
    def accuracy(self,preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return metrics.accuracy_score(pred_flat, labels_flat)    
        
        
        
    def Trainer(self,train,val,epochs):
        
        
        total_steps = len(train) * epochs        
        scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                            num_warmup_steps = 0, #default
                                            num_training_steps = total_steps)


    
        training_stats= {"epoch": [],
                       "train_loss":[],
                       'val_Loss': [],
                       'val_Accur': [] } 
        
        
        for epoch in range(0, epochs):
            loss_per_epoch = 0 
            running_loss= 0 
            self.model.train()
            iteration = 0
            for step, batch in enumerate(train):
                iteration += 1
                b_input_ids = batch[0].to(self.device) 
                b_input_mask = batch[1].to(self.device) 
                b_labels = batch[2].to(self.device) 
    
                loss, logits = self.model(b_input_ids, 
                               token_type_ids=None, 
                               attention_mask=b_input_mask, 
                               labels=b_labels)
             #   print(logits)
                loss_per_epoch += loss.item()
                running_loss += loss.item()
                loss.backward()
                        # This is to help prevent the "exploding gradients" problem
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
                if not step%25:
                    print(f'Epoch: {epoch+1:03d}/{epochs:03d} | '
                          f'Batch {step+1:03d}/{len(train):03d} | '
                          f'Average Loss in last {iteration} iterations: {(running_loss/iteration):.4f}')
                    running_loss = 0
                    iteration = 0
        #torch.cuda.empty_cache() #memory
        #gc.collect() #memory
        #losses.append(float(loss.item()))
                
                
                
                
             
            avg_train_loss = loss_per_epoch / len(train) 
            training_stats["epoch"].append(epoch)
            training_stats["train_loss"].append(avg_train_loss)
            
       
            
            
        for epoch_i in range(0, epochs):
            total_accuracy = 0
            total_val_loss = 0
            with torch.no_grad():
                self.model.eval()
                for batch in val:
        
                    b_input_ids = batch[0].to(self.device) 
                    b_input_mask = batch[1].to(self.device) 
                    b_labels = batch[2].to(self.device) 
                
                    loss, logits = self.model(b_input_ids,token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
                    
                    total_val_loss += loss.item()

        
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()

                    total_accuracy += self.accuracy(logits, label_ids)
               #  print(total_eval_accuracy)
        
  
            avg_accuracy = total_accuracy / len(val) 
            avg_loss = total_val_loss / len(val)
            training_stats["val_Loss"].append(avg_loss)
            training_stats["val_Accur"].append(avg_accuracy)
                        
        return training_stats
        

   

    
    def Test(self,data):
        
     
        test_stats=[]    
        total_accuracy = 0
        total_loss = 0
        predicts = []
        labels= []
            
        self.model.eval()
        with torch.no_grad(): 
             for batch in data:
        
                 b_input_ids = batch[0].to(self.device) 
                 b_input_mask = batch[1].to(self.device) 
                 b_labels = batch[2].to(self.device) 
                
                 loss, logits = self.model(b_input_ids,token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
                    
                 total_loss += loss.item()

        
                 logits = logits.detach().cpu().numpy()
                 label_ids = b_labels.to('cpu').numpy()

                 total_accuracy += self.accuracy(logits, label_ids)
                 predicts.append(logits)
                 labels.append(label_ids)
                 
                    
        predictions = np.concatenate(predicts, axis=0)
        predictions = np.argmax(predictions, axis=1).flatten()
        labels = np.concatenate(labels, axis=0)
   
 
        
  
        avg_accuracy = total_accuracy / len(data)


   
        avg_loss = total_loss / len(data)
    
    

       
  
        test_stats.append(
        {
            ' Loss': avg_loss,
            ' Accur': avg_accuracy})
    
        return test_stats, predictions,labels

    

    