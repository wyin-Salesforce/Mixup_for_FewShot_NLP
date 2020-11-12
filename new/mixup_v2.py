import torch
from scipy.stats import beta
from torch.nn import CrossEntropyLoss


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.cuda.LongTensor(torch.cat([init_dim * torch.arange(n_tile, device=a.device) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def mixup_layer(hidden_states_batch, ele_wise_v1, ele_wise_v2, labels, num_labels, lambda_value, classification_function, is_train=True, use_mixup=True):
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
            batch_size = hidden_states_batch.shape[0]
            '''mix representations'''
            hidden_states_single_v1 = hidden_states_batch.repeat(batch_size, 1)
            hidden_states_single_v2 = tile(hidden_states_batch, 0, batch_size)#torch.repeat_interleave(hidden_states_batch, repeats=batch_size, dim=0)
            combined_pairs = lambda_value*(ele_wise_v1*hidden_states_single_v1)+(1.0-lambda_value)*(ele_wise_v2*hidden_states_single_v2) #(batch*batch, hidden)
            logits = classification_function(combined_pairs) #(batch, tag_set)


            loss_fct = CrossEntropyLoss()

            '''mixup labels'''
            label_ids_v1 = labels.repeat(batch_size)
            label_ids_v2 = torch.repeat_interleave(labels.view(-1, 1), repeats=batch_size, dim=0)

            '''mixup loss'''
            loss_v1 = loss_fct(logits.view(-1, num_labels), label_ids_v1.view(-1))
            loss_v2 = loss_fct(logits.view(-1, num_labels), label_ids_v2.view(-1))
            loss = lambda_value*loss_v1+(1.0-lambda_value)*loss_v2# + 1e-3*reg_loss
            return loss
        else:
            logits = classification_function(hidden_states_batch) #(batch, tag_set)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            return loss
    else: # testing
        logits = classification_function(hidden_states_batch) #(batch, tag_set)
        return logits
