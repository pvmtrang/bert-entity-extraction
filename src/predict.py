import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel
import pandas as pd

import os #get path file name

def process_data(path, enc_tag): #read into a df and convert it bla bla
    data = []
    sentence_cnt = 0
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if not line: #empty line to separate two sentences
                sentence_cnt += 1
            else: 
                line = line.split()
                sentence_word = [sentence_cnt] #sentence_word = [sentence_#, word, tag]
                if len(line) == 4:
                    if line[1] != "_": #huhuhuhu hmm this is the id line
                        # sentences_id.append(" ".join(line)) #hmm hmm
                        pass
                    else:
                        sentence_word.extend([line[0], line[-1]])
                        data.append(sentence_word)

    df = pd.DataFrame(data, columns = ["Sentence #", "Word", "Tag"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, tag

#auto create another file test_name.txt in the same data folder
def write_to_file(sentences, predictions, target_tags): #sentences: list of lists of words, predictions: list of lists of tags
    if (len(sentences) != len(predictions) or (len(sentences) != len(target_tags))):
        raise Exception("Total count of test sentences != of predictions or != target tags")
    output_data = []
    for sentence_cnt in range (len(sentences)):
        print("------i=" + str(sentence_cnt))
        combo = zip(sentences[sentence_cnt], target_tags[sentence_cnt], predictions[sentence_cnt])
        combo_list = list(combo)
        print(combo_list)
        output_data.append(combo_list)

    file_name = os.path.basename(config.TEST_FILE).split(".")[0] + ".txt"
    file_path = os.path.dirname(os.path.abspath(config.TEST_FILE)) #hope this helps
    file_path = os.path.join(file_path, file_name)

    with open(file_path, "w", encoding="utf-8") as file:
        for sentence_cnt in range (len(output_data)):
            for combo in output_data[sentence_cnt]:
                file.write(combo[0] + " _ " + combo[1] + " " + combo[2] + "\n")    #word _ target prediction
            file.write("\n")




if __name__ == "__main__":

    meta_data = joblib.load(config.META_DATA_PATH)
    enc_tag = meta_data["train_enc_tag"]

    num_tag = len(list(enc_tag.classes_))

    sentences, tags = process_data(config.TEST_FILE, enc_tag)
    print("done process data")
    # sentence = """
    # abhishek is going to india
    # """
    # print(tags)

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    tokenized_sentence = []

    for sentence in sentences:
        sentence = config.TOKENIZER.encode(sentence)
        tokenized_sentence.append(sentence)


    test_dataset = dataset.EntityDataset(
        texts=sentences, 
        tags= tags
    )
    print("done loading dataset")


    # for i in range(test_dataset.__len__):
    #     test_dataset.__getitem__(i).to(device)

    count = 0
    output = []
    with torch.no_grad():
        for data in test_dataset:
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            tag, _ = model(**data)

            output.append(
                enc_tag.inverse_transform(
                    tag.argmax(2).cpu().numpy().reshape(-1)
                )[1: (len(tokenized_sentence[count]) - 1)].tolist()
            )

            count += 1
            if(count % 1000 == 0):
                print(count)
            # print("\n")

# print(sentences)
# print(output)
            
# print(enc_tag.inverse_transform([5, 11, 12]))

inverse_tag = []
for tag in tags:
    # print("----")
    # print(tag)
    tag = enc_tag.inverse_transform(np.array(tag))
    # print(tag)
    inverse_tag.append(tag)

print("inverse_tag")

write_to_file(sentences, output, target_tags = inverse_tag)
