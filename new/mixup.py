import torch
from scipy.stats import beta
from torch.nn import CrossEntropyLoss

def mixup_layer(hidden_states_batch, labels, num_labels, lambda_value, classification_function, is_train=True, use_mixup=True):
    '''
    Inputs:
        hidden_states_batch: (batch_size, hidden_size), representations for input batch
        labels: (batch_size), a list of gold labels
        num_labels: the total label size
        lambda_value: a random value in beta sampling
        classification_function: A function from hidden representation to gold labels, such as "RobertaClassificationHead(bert_hidden_dim, tagset_size)"
        is_train: True or False. It turns mixup off in testing, and
    Training Outputs:
        loss: a scalar
    Testing Outputs:
        logits: (batch_size, label_size), before softmax
    '''
    if is_train is False and use_mixup:
        print('Error, you cannot use mixup in testing')
        exit(0)
    if is_train:
        if use_mixup:
            batch_size = hidden_states_batch.shape[0]#.cpu().numpy()
            print('hidden_states_batch:', hidden_states_batch)
            '''mix representations'''
            hidden_states_single_v1 = hidden_states_batch.repeat(batch_size, 1)
            hidden_states_single_v2 = torch.repeat_interleave(hidden_states_batch, repeats=batch_size, dim=0)
            combined_pairs = lambda_value*hidden_states_single_v1+lambda_value*hidden_states_single_v1 #(batch*batch, hidden)
            print('combined_pairs:', combined_pairs)
            logits = classification_function(combined_pairs) #(batch, tag_set)


            loss_fct = CrossEntropyLoss()

            '''mixup labels'''
            label_ids_v1 = labels.repeat(batch_size)
            label_ids_v2 = torch.repeat_interleave(labels.view(-1, 1), repeats=batch_size, dim=0)

            '''mixup loss'''
            loss_v1 = loss_fct(logits.view(-1, num_labels), label_ids_v1.view(-1))
            loss_v2 = loss_fct(logits.view(-1, num_labels), label_ids_v2.view(-1))
            loss = lambda_value*loss_v1+lambda_value*loss_v2# + 1e-3*reg_loss
            return loss
        else:
            logits = classification_function(hidden_states_batch) #(batch, tag_set)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            return loss
    else: # testing
        logits = classification_function(hidden_states_batch) #(batch, tag_set)
        return logits
