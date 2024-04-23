import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel

if __name__ == "__main__":
    enc_tag = preprocessing.LabelEncoder()

    train_dataset = dataset.EntityDataset(config.TRAINING_FILE, enc_tag)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4, shuffle = True
    )

    valid_dataset = dataset.EntityDataset(config.VALIDATION_FILE, enc_tag)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1, shuffle = True
    )

    num_tag_train = len(list(train_dataset.get_enc_tag().classes_))
    num_tag_val = len(list(valid_dataset.get_enc_tag().classes_))

    if (num_tag_train != num_tag_val):
        raise Exception("Number of tags in Train != Val OMG")
    
    enc_tag = train_dataset.get_enc_tag()
    
    meta_data = {
        "enc_tag": enc_tag,
    }
    joblib.dump(meta_data, config.META_DATA_PATH)
    
    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag_train, enc_tag=enc_tag)
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

    num_train_steps = int(train_dataset.__len__() / config.TRAIN_BATCH_SIZE * config.EPOCHS) #hmm, should i use __len__?
    optimizer = AdamW(optimizer_parameters, lr=3e-6) #original: 3e-5
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    print(config.CURRENT_BIN_FOLDER)

    best_f1 = - np.inf
    for epoch in range(config.EPOCHS):
        print("----Epoch #" + str(epoch + 1))
        train_loss, train_f1 = engine.train_fn(train_data_loader, model, optimizer, scheduler, device)
        val_loss, val_f1 = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train F1 = {train_f1} Train Loss = {train_loss} Valid F1 = {val_f1} Valid Loss = {val_loss}")
        if val_f1 > best_f1:
            print("new best model")
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_f1 = val_f1
    print("done\nbest val_f1: " + str(best_f1))