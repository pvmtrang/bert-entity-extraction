import config
import torch
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset

def process_data(path, enc_tag): #read into a df and convert it into 2 lists: sentence, tag. and enc_tag
    data = []
    sentence_cnt = 0
    with open(path, encoding="utf8") as file:
        for line in file:
            #print(line)
            line = line.strip()
            if not line: #empty line to separate two sentences
                sentence_cnt += 1
            else: 
                line = line.split()
                sentence_word = [sentence_cnt] #sentence_word = [sentence_#, word, tag]
                if len(line) != 4 or line[1] != "_": #test
                    #print(line)
                    continue
                else:
                    sentence_word.extend([line[0], line[-1]])
                    data.append(sentence_word)
    df = pd.DataFrame(data, columns = ["Sentence #", "Word", "Tag"])

    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, tag, enc_tag


class EntityDataset(Dataset):
    def __init__(self, path, enc_tag = None):
        if enc_tag:
            self.enc_tag = enc_tag
        else:
            self.enc_tag = preprocessing.LabelEncoder()
        self.sentences, self.tags, self.enc_tag = process_data(path, enc_tag=self.enc_tag)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, item):
        text = self.sentences[item]
        ids = []

        target_tag =[]
        tags = self.tags[item]
        real_tokenized_len = 0
        
        for i, s in enumerate(text): #for each word in seq
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            # abhishek: ab ##hi ##sh ##ek
            input_len = len(inputs)
            real_tokenized_len += input_len
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)
        

        ids = ids[:config.MAX_LEN - 2]
        ids = [101] + ids + [102]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        padding_len = config.MAX_LEN - len(ids)
        
        target_tag = target_tag[:config.MAX_LEN - 2]
        target_tag = [0] + target_tag + [0]
        
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        
        return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long), #which sentence does this token belong to 
                "target_tag": torch.tensor(target_tag, dtype=torch.long),
                "real_tokenized_len": torch.tensor(real_tokenized_len, dtype=torch.long)
            }

    def get_enc_tag(self):
        return self.enc_tag
    
    def get_sentences(self):
        return self.sentences
    
    def get_tags(self):
        target_tag = []
        for sentence_id in range(len(self.tags)):
            target_tag.append(self.__getitem__(sentence_id)['target_tag'])
        return target_tag
