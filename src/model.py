import config
import torch
import transformers
import torch.nn as nn
import utils
import conlleval

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1    #reshape into one 1_d array and which element is not the padding 0
    active_logits = output.view(-1, num_labels) #divide into n columns wtf is this for
    active_labels = torch.where(        #if not padding -> target, else -100.0
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)  #-100.0
    )
    loss = lfn(active_logits, active_labels)
    # print(loss)
    return loss

def f1_fn(tag, target_tag, enc_tag, real_tokenized_len, ids, is_test_mode):
    real_tokenized_len = [sen.item() for sen in real_tokenized_len] #extract len of each tokenized sentence -> list of len
    # print("real_tokenized len")
    # print(real_tokenized_len)
    pred = tag.argmax(2) #tag id for max_len tokens. [batch_size, max_len]
    pred_tags, true_tags = [], [] #a list of tensor
    for i, _ in enumerate(zip(pred, target_tag, real_tokenized_len)):
        pred_tags.append(pred[i][1:real_tokenized_len[i]+1])
        true_tags.append(target_tag[i][1:real_tokenized_len[i]+1])
        
    inverse_pred_tags, inverse_true_tags = [], []
    for id in range(len(pred_tags)):
        inverse_pred_tags.append(enc_tag.inverse_transform(pred_tags[id].cpu().numpy().reshape(-1)))
        inverse_true_tags.append(enc_tag.inverse_transform(true_tags[id].cpu().numpy().reshape(-1)))
    # print(inverse_true_tags)
    
    tokenized_sentences = []
    for id in ids:
        tokenized_sentences.append((config.TOKENIZER.convert_ids_to_tokens(id, skip_special_tokens = True)))

    if is_test_mode: #a list of lists, 2 lists of np arrays?? or sth whatever
        return tokenized_sentences, inverse_true_tags, inverse_pred_tags
    else:
        output_file_path = "output/batch.tmp"
        utils.write_to_file(tokenized_sentences, inverse_pred_tags, inverse_true_tags, output_file_path)
        res = conlleval.evaluate_conll_file(output_file_path, False)
        # print(res)
        return res
        

class EntityModel(nn.Module):
    def __init__(self, num_tag, enc_tag, is_test_mode = False):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.enc_tag = enc_tag
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH,return_dict=False)
        self.bert_drop_1 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag) #768: the hidden size of bert base
        self.is_test_mode = is_test_mode
    
    #token_type_ids: Sentence embedding: E_A, E_B...
    def forward(self, ids, mask, token_type_ids, target_tag, real_tokenized_len):
        # text = config.TOKENIZER.batch_decode(ids, skip_special_tokens=True)
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids) #o1: embedding for each token

        bo_tag = self.bert_drop_1(o1)
        tag = self.out_tag(bo_tag)
        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)         
        if self.is_test_mode:
            tokenized_sentences, inverse_true_tags, inverse_pred_tags = f1_fn(tag, target_tag, self.enc_tag, real_tokenized_len, ids, self.is_test_mode)
            return tokenized_sentences, inverse_true_tags, inverse_pred_tags
        else:
            f1_res = f1_fn(tag, target_tag, self.enc_tag, real_tokenized_len, ids, self.is_test_mode)
            return tag, loss_tag, f1_res
            

        
