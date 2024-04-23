import joblib
import torch
import config
import dataset
import utils
import conlleval

import os
from model import EntityModel
import engine

if __name__ == "__main__":

    meta_data = joblib.load(config.META_DATA_PATH)
    enc_tag = meta_data["enc_tag"]
    data_file_name = os.path.basename(config.VALIDATION_FILE).split(".")[0] + ".txt"
    output_file_path = config.VALIDATION_OUTPUT_FILE + data_file_name

    test_dataset = dataset.EntityDataset(config.VALIDATION_FILE, enc_tag)
    num_tag = len(list(enc_tag.classes_))

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=4
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag, enc_tag=enc_tag, need_f1_report=True)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    tokenized_sentences, inverse_pred_tags, inverse_true_tags = engine.test_fn(test_data_loader, model, device)
    
    all_sentences, all_pred_tags, all_true_tags = utils.combine_subwords(tokenized_sentences, inverse_pred_tags, inverse_true_tags)
    # utils.write_to_file(all_sentences, all_pred_tags, all_true_tags, output_file_path)
    utils.write_to_file(tokenized_sentences, inverse_pred_tags, inverse_true_tags, output_file_path)
    conlleval.evaluate_conll_file(output_file_path, verbose=True)
    # print("\n")
    # conlleval.evaluate(all_true_tags, all_pred_tags)
