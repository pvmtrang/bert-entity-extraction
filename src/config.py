import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "../input/bert-base-uncased"
# https://huggingface.co/google-bert/bert-base-uncased
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/ner_dataset.csv"
# https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus?resource=download&select=ner_dataset.csv
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    pretrained_model_name_or_path = BASE_MODEL_PATH,
    do_lower_case=True
)
