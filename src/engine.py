import torch
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, scheduler, device = torch.device('cuda')):
    model.train()
    final_loss = 0
    final_f1 = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad() #clear out all the previously tracked gradients
        _, loss, f1_res = model(**data) 
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item() 
        final_f1 += f1_res[2]
    
    return final_loss / len(data_loader), final_f1 / len(data_loader) 


def eval_fn(data_loader, model, device = torch.device("cuda")):
    model.eval()
    final_loss = 0
    final_f1 = 0
    with torch.no_grad(): #reduct mem consumption coz we dont use .backward() here
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
            _, loss, f1_res = model(**data)
            final_loss += loss.item()
            final_f1 += f1_res[2]
    return final_loss / len(data_loader), final_f1 / len(data_loader) 

def test_fn(data_loader, model, device = torch.device("cuda")):
    tokenized_sentences = []
    inverse_pred_tags = []
    inverse_true_tags = []

    model.eval() #turn off dropouts, etc

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
            tokenized_sentence, inverse_pred_tag, inverse_true_tag = model(**data)

            tokenized_sentences.extend(tokenized_sentence)
            inverse_pred_tags.extend(inverse_pred_tag)
            inverse_true_tags.extend(inverse_true_tag)
    return tokenized_sentences, inverse_pred_tags, inverse_true_tags
