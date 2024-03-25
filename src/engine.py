import torch
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    final_f1 = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss, f1_res = model(**data) 
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item() 
        final_f1 += f1_res[2]
    
    return final_loss / len(data_loader), final_f1 / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    final_f1 = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        _, loss, f1_res = model(**data)
        final_loss += loss.item()
        final_f1 += f1_res[2]
    return final_loss / len(data_loader), final_f1 / len(data_loader) 
