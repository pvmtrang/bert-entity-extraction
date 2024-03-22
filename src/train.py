import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


def process_data(path): #read into a df and convert it into 2 lists: sentence, tag. and enc_tag
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

    #df = df.loc[:500] #test

    #print("-- Set of Tags: " + " ".join(df['Tag'].unique()) + " -> " +str(df['Tag'].nunique()))

    enc_tag = preprocessing.LabelEncoder()
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, tag, enc_tag


if __name__ == "__main__":
    train_sentences, train_tag, train_enc_tag = process_data(config.TRAINING_FILE)
    # print("done train set")
    val_sentences, val_tag, val_enc_tag = process_data(config.VALIDATION_FILE)

    # print("train_sentences = " + str(len(train_sentences)))
    # print("val_sentences = " + str(len(val_sentences)))

    meta_data = { #sua lai cai nay, 1 enc_tag thoi 
        "train_enc_tag": train_enc_tag,
        "val_enc_tag": val_enc_tag,
    }
    joblib.dump(meta_data, config.META_DATA_PATH)

    num_tag_train = len(list(train_enc_tag.classes_)) #du sao thi train total tag == val == test total tag
    num_tag_val = len(list(val_enc_tag.classes_)) #du sao thi train total tag == val == test total tag

    if (num_tag_train != num_tag_val):
        raise Exception("Number of tags in Train != Val OMG")

    train_dataset = dataset.EntityDataset(
        texts=train_sentences, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4, shuffle = True
    )

    valid_dataset = dataset.EntityDataset(
        texts=val_sentences, tags=val_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1, shuffle = True
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag_train)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        val_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {val_loss}")
        if val_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = val_loss