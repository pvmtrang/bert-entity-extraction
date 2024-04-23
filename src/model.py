import config
import transformers
import torch.nn as nn
import utils
       

class EntityModel(nn.Module):
    def __init__(self, num_tag, enc_tag, need_f1_report = False):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.enc_tag = enc_tag
        if config.MODEL_NAME == 'bert':
            self.model = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH, return_dict = False)
        elif config.MODEL_NAME == 'xlm-roberta':   
            self.model = transformers.XLMRobertaModel.from_pretrained(config.BASE_MODEL_PATH, return_dict = False)
        self.dropout = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag) #768: the hidden size of bert base and roberta also
        self.need_f1_report = need_f1_report #huhu i have no other name. 
    
    #token_type_ids: Sentence embedding: E_A, E_B...
    def forward(self, ids, mask, token_type_ids, target_tag, real_tokenized_len):
        # text = config.TOKENIZER.batch_decode(ids, skip_special_tokens=True)
        o1, _ = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids) #o1: embedding for each token
        o_tag = self.dropout(o1)
        tag_logits = self.out_tag(o_tag)
        loss_tag = utils.loss_fn(tag_logits, target_tag, mask, self.num_tag)         
        if self.need_f1_report:
            tokenized_sentences, inverse_pred_tags, inverse_true_tags = utils.f1_fn(tag_logits, target_tag, self.enc_tag, real_tokenized_len, ids, self.need_f1_report)
            return tokenized_sentences, inverse_pred_tags, inverse_true_tags
        else:
            f1_res = utils.f1_fn(tag_logits, target_tag, self.enc_tag, real_tokenized_len, ids, self.need_f1_report)
            return tag_logits, loss_tag, f1_res
            

        
