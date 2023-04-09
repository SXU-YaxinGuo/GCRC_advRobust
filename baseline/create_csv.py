import csv
import json
import tqdm

def create_csv_test_public():
    header = ['id','title','passage','question','options','positive_options','negative_question','negative_options']
    sample_list = []
    with open("./data/advRobust_data/test_GCRCadvrobust_public.json", encoding="utf-8") as f:
        data = json.load(f)
        for k in tqdm.tqdm(data["data"]):
            sample = (k["id"],k['title'],k['passage'],k['question'],k['options'],k["positive_options"],k['negative_question'],k['negative_options'])
            sample_list.append(sample)
            with open("./data/advRobust_data/test_GCRCadvrobust_public.csv","w",encoding='utf-8',newline='') as b:
                writer = csv.writer(b)
                writer.writerow(header)
                for p in sample_list:
                    writer.writerow(p)

def create_csv_dev():
    header = ['id','title','passage','question','options','answer','positive_options','positive_answer','negative_question','negative_options','negative_answer']
    sample_list = []
    with open("./data/advRobust_data/dev_GCRCadvrobust.json", encoding="utf-8") as f:
        data = json.load(f)
        for k in tqdm.tqdm(data["data"]):
            sample = (k["id"],k['title'],k['passage'],k['question'],k['options'],k['answer'],k["positive_options"],k['positive_answer'],k['negative_question'],k['negative_options'],k['negative_answer'])
            sample_list.append(sample)
            with open("./data/advRobust_data/dev_GCRCadvrobust.csv","w",encoding='utf-8',newline='') as b:
                writer = csv.writer(b)
                writer.writerow(header)
                for p in sample_list:
                    writer.writerow(p)

def create_csv_train():
    header = ['id','title','passage','question','options','answer']
    sample_list = []
    with open("./data/advRobust_data/train_GCRC.json", encoding="utf-8") as f:
        data = json.load(f)
        for k in tqdm.tqdm(data["data"]):
            sample = (k["id"],k['title'],k['passage'],k['question'],k['options'],k['answer'])
            sample_list.append(sample)
            with open("./data/advRobust_data/train_GCRC.csv","w",encoding='utf-8',newline='') as b:
                writer = csv.writer(b)
                writer.writerow(header)
                for p in sample_list:
                    writer.writerow(p)
create_csv_test_public()
