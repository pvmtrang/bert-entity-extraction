import transformers

MAX_LEN = 128
# reduce batch size because dataset size << original. Original train: 43613 sentences, coner: 15300. orig val: 4796, coner: 800
TRAIN_BATCH_SIZE = 10 #from 32 -> 10
VALID_BATCH_SIZE = 3 #from 8 -> 3
EPOCHS = 10
BASE_MODEL_PATH = "../input/bert-base-uncased"
# https://huggingface.co/google-bert/bert-base-uncased
MODEL_PATH = "bin/coner2022_Mar15/model.bin"
META_DATA_PATH = "bin/coner2022_Mar15/meta.bin"
#TRAINING_FILE = "../input/ner_dataset.csv"
TRAINING_FILE = "../input/multiconer1_2022/EN-English/en_train.conll"
VALIDATION_FILE = "../input/multiconer1_2022/EN-English/en_dev.conll"
TEST_FILE = "../input/multiconer1_2022/EN-English/en_test.conll"
# https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus?resource=download&select=ner_dataset.csv
# multiconer 1: aws s3 cp --no-sign-request s3://multiconer/multiconer2022/ multiconer2022/ --recursive
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    pretrained_model_name_or_path = BASE_MODEL_PATH,
    do_lower_case=True
)

# USING_CONLL = True
