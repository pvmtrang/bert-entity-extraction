import joblib
import torch
import config
import dataset
import utils
import conlleval

import os
from model import EntityModel
from tqdm import tqdm

if __name__ == "__main__":

    meta_data = joblib.load(config.META_DATA_PATH)
    enc_tag = meta_data["enc_tag"]
    data_file_name = os.path.basename(config.TEST_FILE).split(".")[0] + ".txt"
    output_file_path = "output/" + data_file_name

    test_dataset = dataset.EntityDataset(config.TEST_FILE, enc_tag)
    num_tag = len(list(enc_tag.classes_))

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=4
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag, enc_tag=enc_tag, is_test_mode=True)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    tokenized_sentences = []
    inverse_pred_tags = []
    inverse_true_tags = []

    with torch.no_grad():
        for data in tqdm(test_data_loader, total=len(test_data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
            tokenized_sentence, inverse_true_tag, inverse_pred_tag = model(**data)
            tokenized_sentences.extend(tokenized_sentence)
            inverse_pred_tags.extend(inverse_true_tag)
            inverse_true_tags.extend(inverse_true_tag)
    
    utils.write_to_file(tokenized_sentences, inverse_pred_tags, inverse_pred_tag, output_file_path)
    conlleval.evaluate_conll_file(output_file_path, verbose=True)
