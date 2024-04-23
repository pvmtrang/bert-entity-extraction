import transformers

MODEL_NAME = 'bert'
# MODEL_NAME = 'xlm-roberta'

MAX_LEN = 64 #the longest sentence in test, train, dev set is 49 words long
TRAIN_BATCH_SIZE = 32 
VALID_BATCH_SIZE = 8 
TEST_BATCH_SIZE = 256
EPOCHS = 30
if MODEL_NAME == 'bert':
    BASE_MODEL_PATH = "../input/model/bert-base-uncased"
elif MODEL_NAME == 'xlm-roberta':
    BASE_MODEL_PATH = "../input/model/xlm-roberta-base"

CURRENT_BIN_FOLDER = "bin/coner2022_Apr23_bert_lr3e6/"
# https://huggingface.co/google-bert/bert-base-uncased
# CURRENT_BIN_FOLDER = "bin/coner2022_Apr20_xlm-roberta/"

MODEL_PATH = CURRENT_BIN_FOLDER + "model.bin"
META_DATA_PATH = CURRENT_BIN_FOLDER + "meta.bin"
TRAINING_OUTPUT_FILE = CURRENT_BIN_FOLDER + "train-"
VALIDATION_OUTPUT_FILE = CURRENT_BIN_FOLDER + "val-"
VALIDATION_BEST_OUTPUT_FILE = CURRENT_BIN_FOLDER + "val-best-"
TEST_OUTPUT_FILE = CURRENT_BIN_FOLDER + "test-"

TRAINING_FILE = "../input/data/multicon er1_2022/EN-English/en_train.conll"
VALIDATION_FILE = "../input/data/multiconer1_2022/EN-English/en_dev.conll"
TEST_FILE = "../input/data/multiconer1_2022/EN-English/en_test.conll"
# https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus?resource=download&select=ner_dataset.csv
# multiconer 1: aws s3 cp --no-sign-request s3://multiconer/multiconer2022/ multiconer2022/ --recursive

if MODEL_NAME == 'bert': 
    TOKENIZER = transformers.BertTokenizer.from_pretrained(
        pretrained_model_name_or_path = BASE_MODEL_PATH,
        do_lower_case=True
    )
elif MODEL_NAME == 'xlm-roberta':    
    TOKENIZER = transformers.XLMRobertaTokenizer.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL_PATH
    )

