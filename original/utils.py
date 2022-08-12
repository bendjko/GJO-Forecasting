import pandas as pd
import datetime
from transformers import BertTokenizerFast
import pickle
import os
import torch
from torch.utils.data import DataLoader
import random
from sklearn.metrics import classification_report
from statsmodels.stats.contingency_tables import mcnemar
import spacy
spacy_eng = spacy.load('en_core_web_sm')
def tokenize(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
# def round(num):
#     decimal = num % 1
#     if decimal >= 0.5:
#         return (num - decimal) + 1
#     else:
#         return num - decimal
class Queue:
    def __init__(self, size, title, special_token):
        self.q = []
        self.size = size
        self.index = 0
        self.title = title
        self.special = special_token
    def add(self, val):
        self.q.append(val)
        self.limit()
    def limit(self):
        if len(self.q) > self.size:
            self.index +=1
    def get_list_days(self):
        return self.q[self.index: ]
    def get_list_forecasts(self):
        forecasts = {}
        flags = {}
        last_day = self.q[len(self.q)-1]
        for day in self.q[self.index: ]:
            for forecast in day:
                if forecast.user_id in forecasts:
                    del forecasts[forecast.user_id]
                forecasts[forecast.user_id] = forecast
                if forecast.user_id in flags:
                    del flags[forecast.user_id]
                if day == last_day:
                    # print("Entered")
                    flags[forecast.user_id] = 0
                else:
                    flags[forecast.user_id] = 1
        return list(forecasts.values()), list(flags.values())
    def get_baseline_predictions(self):
        forecasts, flags = self.get_list_forecasts()
        majority_predictions = []
        weighted_predictions = []
        for forecast in forecasts:
            majority_predictions.append(round(forecast.forecast_prediction))
            weighted_predictions.append(forecast.forecast_prediction)
        maj_pred = int(sum(majority_predictions) / len(majority_predictions) >= 0.5)
        weight_pred = round(sum(weighted_predictions) / len(weighted_predictions))
        return maj_pred, weight_pred
    def get_input(self):
        forecasts, flags = self.get_list_forecasts()
        text = []
        predictions = []
        for forecast in forecasts:
            text.append(forecast.text)
            predictions.append(forecast.forecast_prediction)
        return text, predictions, flags
class Question:
    def __init__(self, path, tokenizer):
        data = pd.read_json(path)
        self.id = data["question_id"][0]
        self.title = data["title"][0]
        self.correct_answer = int(data["correct_answer"][0] == "Yes")
        self.crowd_forecast = data["crowd_forecast"][0]
        self.df = pd.DataFrame(data["preds"].tolist(), columns=["user_id", "timestamp", "forecast", "text"])
        self.df["question_id"] = self.id
        self.tokenizer = tokenizer
        self.special_token = self.tokenizer.sep_token
        self.total_days = []

        self.input_ids = []
        self.attention_masks = []
        self.forecast_predictions = []
        self.correct_answer_list = []
        self.question_input = []
        self.question_attention = []

        self.question_encoding = self.tokenizer(self.title, padding="max_length", truncation=True, add_special_tokens=True)
        self.question_input_id = self.question_encoding["input_ids"]
        self.question_attention_mask = self.question_encoding["attention_mask"]
        self.text_2_encoding = {}

        self.flags = []

    def __repr__(self):
        return f"Question ID: {self.id}\nTitle: {self.title}\nCorrect Answer: {self.correct_answer}\nCrowd Forecast: {self.crowd_forecast}"
    def setDaily(self, daily):
        self.daily = daily
    def __getitem__(self, ind):
        return self.df.iloc[ind]

    def __len__(self):
        return len(self.df)
    def build_all_encodings(self):
        unique_texts = []
        for day in self.total_days:
            for forecast in day:
                if forecast.text not in unique_texts:
                    unique_texts.append(forecast.text)
        all_encodings = self.tokenizer(unique_texts, padding="max_length", truncation=True, add_special_tokens=True)
        for i in range(len(unique_texts)):
            d = {'input_ids': all_encodings["input_ids"][i],
                 'attention_mask': all_encodings["attention_mask"][i]}
            self.text_2_encoding[unique_texts[i]] = d

    def build_input(self):
        total_values = Queue(10, self.title, self.special_token)
        self.flags = []
        for day in self.total_days:
            daily_text = []
            daily_predictions = []
            daily_question_input_ids = []
            daily_question_attention_mask = []
            total_values.add(day)
            for forecast in day:
                daily_text.append(forecast.text)
                daily_predictions.append(forecast.forecast_prediction)
                daily_question_input_ids.append(self.question_input_id)
                daily_question_attention_mask.append(self.question_attention_mask)
            if self.daily:
                encoding_daily = self.tokenizer(daily_text, padding="max_length", truncation=True,
                                                add_special_tokens=True)
                self.input_ids.append(encoding_daily["input_ids"])
                self.attention_masks.append(encoding_daily["attention_mask"])
                self.forecast_predictions.append(daily_predictions)
                self.question_input.append(daily_question_input_ids)
                self.question_attention.append(daily_question_attention_mask)
            # print('Entered daily')

            else:
                day_input_ids = []
                day_attention_masks = []
                day_question_input = []
                day_question_attention = []
                total_text, total_predictions, flags = total_values.get_input()
                for text in total_text:
                    day_input_ids.append(self.text_2_encoding[text]["input_ids"])
                    day_attention_masks.append(self.text_2_encoding[text]["attention_mask"])
                    day_question_input.append(self.question_input_id)
                    day_question_attention.append(self.question_attention_mask)
                self.input_ids.append(day_input_ids)
                self.attention_masks.append(day_attention_masks)
                self.forecast_predictions.append(total_predictions)
                self.flags.append(flags)
                self.question_input.append(day_question_input)
                self.question_attention.append(day_question_attention)
            # print("Entered total")
            self.correct_answer_list.append(self.correct_answer)
        # length = len(self.input_ids)
        # self.input_ids = self.input_ids[:int(length/4)]
        # self.attention_masks = self.attention_masks[:int(length/4)]
        # self.forecast_predictions = self.forecast_predictions[:int(length/4)]
        # self.question_input = self.question_input[:int(length/4)]
        # self.question_attention = self.question_attention[:int(length/4)]
        # self.correct_answer_list = self.correct_answer_list[:int(len(self.correct_answer_list)/4)]

    def build_day_lists(self):
        days = []
        for index, row in self.df.iterrows():
            forecast = Forecast(row["user_id"], row["timestamp"], row["text"], row["forecast"], self.id, self.title)
            if len(days) == 0:
                days.append(forecast)
            else:
                last_date = days[len(days)-1].date
                if last_date == forecast.date:
                    days.append(forecast)
                else:
                    self.total_days.append(days)
                    days = [forecast]
        self.total_days.append(days)
    def build_all_baselines(self):
        majority_vote_total_dict = {}
        weighted_vote_total_dict = {}

        majority_vote_total = 0
        majority_vote_daily = 0
        weighted_vote_daily = 0
        weighted_vote_total = 0

        y_pred_list_weighted_daily = []
        y_pred_list_maj_daily = []
        y_pred_list_maj_total = []
        y_pred_list_weighted_total = []
        y_correct_list = []
        majority_vote_total_queue = Queue(10, self.title, self.special_token)
        weighted_vote_total_queue = Queue(10, self.title, self.special_token)
        for day in self.total_days:
            majority_daily = 0
            weighted_daily = 0
            majority_vote_total_queue.add(day)
            weighted_vote_total_queue.add(day)
            for forecast in day:
                majority_vote_total_dict[forecast.user_id] = round(forecast.forecast_prediction)
                weighted_vote_total_dict[forecast.user_id] = forecast.forecast_prediction
                majority_daily = majority_daily + round(forecast.forecast_prediction)
                weighted_daily += forecast.forecast_prediction
            baseline_weighted_daily = int(round(weighted_daily / len(day)) == self.correct_answer)
            maj_total_pred, weighted_total_pred = majority_vote_total_queue.get_baseline_predictions()
            baseline_maj_total = int(maj_total_pred == self.correct_answer)
            baseline_weighted_total = int(weighted_total_pred == self.correct_answer)
            baseline_maj_daily = int(int(majority_daily / len(day) >= 0.5) == self.correct_answer)
            y_correct_list.append(self.correct_answer)
            y_pred_list_maj_total.append(maj_total_pred)
            y_pred_list_maj_daily.append(int(majority_daily / len(day) >= 0.5))
            y_pred_list_weighted_daily.append(round(weighted_daily / len(day)))
            y_pred_list_weighted_total.append(weighted_total_pred)
            majority_vote_total += baseline_maj_total
            majority_vote_daily += baseline_maj_daily
            weighted_vote_daily += baseline_weighted_daily
            weighted_vote_total += baseline_weighted_total
        length = len(self.total_days)
        return majority_vote_total, majority_vote_daily, weighted_vote_daily, weighted_vote_total,y_pred_list_maj_total, y_pred_list_maj_daily, y_pred_list_weighted_total, y_pred_list_weighted_daily,y_correct_list, length

    def build_info(self, daily):
        self.setDaily(daily)
        self.build_all_encodings()
        self.build_input()
    def quarters(self, index, results):
        y_correct_list = []
        for i in range(len(self.total_days)):
            y_correct_list.append(self.correct_answer)
        question_results = results[index: index + len(self.total_days)]
        length = len(question_results)
        quarter_results = question_results[int(length/4)*3:]
        quarter_correct = y_correct_list[int(length/4)*3:]
        new_index = index + len(self.total_days)
        return quarter_results,quarter_correct, new_index


class Forecast:
    def __init__(self, user_id, date, text, forecast_prediction, question_id, question_title):
        self.user_id = user_id
        self.date = datetime.datetime.strptime(date[0:10].replace("-", "/"), "%Y/%m/%d")
        self.text = text
        self.forecast_prediction = forecast_prediction
        self.question_id = question_id
        self.title = question_title

    def __lt__(self, other):
        return self.date < other.date
    def __eq__(self, other):
        return (self.user_id , self.date, self.text, self.forecast_prediction, self.question_id, self.title) == (other.user_id, other.date, other.text, other.forecast_prediction, other.question_id, other.title)
    def __hash__(self):
        return hash((self.user_id, self.date, self.text, self.forecast_prediction, self.question_id, self.title))

class GJO_Dataset():
    def __init__(self, questions):
        self.questions = questions
        self.build_input = []
        self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.special_token = self.bert_tokenizer.sep_token

        self.majority_vote_total = 0
        self.majority_vote_daily = 0
        self.weighted_vote_total = 0
        self.weighted_vote_daily = 0

        self.input_ids = []
        self.attention_masks = []
        self.correct_answers = []
        self.forecast_predictions = []
        self.question_input_ids = []
        self.question_attention_masks = []
        self.flags = []
    def build_dataset(self, daily):
        for i in range(len(self.questions)):
            if self.build_input[i]:
                question = self.questions[i]
                question.build_info(daily)
                self.questions[i] = question
                self.input_ids.extend(question.input_ids)
                self.attention_masks.extend(question.attention_masks)
                self.forecast_predictions.extend(question.forecast_predictions)
                self.correct_answers.extend(question.correct_answer_list)
                self.question_input_ids.extend(question.question_input)
                self.question_attention_masks.extend(question.question_attention)
                self.flags.extend(question.flags)
                print("Question completed")
    def quarters(self, fp):
        with open(fp, "rb") as f:
            results = pickle.load(f)
        self.load_dataset("Daily_Total_Special/test.save")
        y_pred = []
        for i in range(len(self.correct_answers)):
            if results[i] == 1:
                y_pred.append(self.correct_answers[i])
            elif results[i] == 0 and self.correct_answers[i]==0:
                y_pred.append(1)
            elif results[i] == 0 and self.correct_answers[i] == 1:
                y_pred.append(0)
        print(classification_report(self.correct_answers, y_pred))
        print(sum(results)/len(results) * 100)
        index = 0
        y_pred_list = []
        y_correct_list = []
        for question in self.questions:
            result, correct, new_index = question.quarters(index, y_pred)
            y_pred_list.extend(result)
            y_correct_list.extend(correct)
            index = new_index
        print(classification_report(y_correct_list, y_pred_list))
        correct = 0
        for i in range(len(y_correct_list)):
            if y_correct_list[i] == y_pred_list[i]:
                correct+=1
        print(correct/len(y_correct_list) * 100)
    def save_quarters(self, load_fp, save_fp):
        with open(load_fp, "rb") as f:
            results = pickle.load(f)
        index = 0
        y_save_list = []
        for question in self.questions:
            total_sub = results[index: index + len(question.total_days)]
            length = len(total_sub)
            quarter_sub = total_sub[int(length/4)*3:]
            y_save_list.extend(quarter_sub)
            index = index + len(question.total_days)
        print(sum(y_save_list)/len(y_save_list) * 100)
        print(sum(results)/len(results) * 100)
        with open(save_fp, "wb") as f:
            pickle.dump(y_save_list, f)
    def do_all_baselines(self):
        count = 0
        y_pred_list_weighted_daily = []
        y_pred_list_maj_daily = []
        y_pred_list_maj_total = []
        y_pred_list_weighted_total = []
        y_correct_list = []
        for question in self.questions:
            maj_total, maj_daily, weighted_daily, weighted_total, l1, l2, l3, l4, l7, f5 = question.build_all_baselines()
            self.majority_vote_daily+=maj_daily
            self.majority_vote_total+=maj_total
            self.weighted_vote_daily+=weighted_daily
            self.weighted_vote_total+=weighted_total
            y_pred_list_maj_total.extend(l1)
            y_pred_list_maj_daily.extend(l2)
            y_pred_list_weighted_total.extend(l3)
            y_pred_list_weighted_daily.extend(l4)
            y_correct_list.extend(l7)
            count+=f5
            self.build_input.append(True)
        self.majority_vote_total /= count #len(self.questions)
        self.majority_vote_daily /= count #len(self.questions)
        self.weighted_vote_total /= count #len(self.questions)
        self.weighted_vote_daily /= count #len(self.questions)
        print("Majority_vote_total", self.majority_vote_total * 100)
        print("Majority_vote_daily", self.majority_vote_daily * 100)
        print("Weighted_vote_daily", self.weighted_vote_daily * 100)
        print("Weighted_vote_total", self.weighted_vote_total * 100)
        print("####################################################################")
        print("Majority_vote_total")
        print(classification_report(y_correct_list, y_pred_list_maj_total))
        print("#####################")
        print("Majority_vote_daily")
        print(classification_report(y_correct_list, y_pred_list_maj_daily))
        print("#####################")
        print("Weighted_vote_daily")
        print(classification_report(y_correct_list, y_pred_list_weighted_daily))
        print("#####################")
        print("Weighted_vote_total")
        print(classification_report(y_correct_list, y_pred_list_weighted_total))
        print("#####################")
        print(len(y_correct_list))

    def load_dataset(self, fp):
        with open(fp, "rb") as f:
            load_dict = pickle.load(f)
            self.input_ids = load_dict['Input_ids']
            self.attention_masks = load_dict['Attention_masks']
            self.forecast_predictions = load_dict['Forecast_predictions']
            self.correct_answers = load_dict['Correct_answers']
            self.question_input_ids = load_dict['Question_input_ids']
            self.question_attention_masks = load_dict['Question_attention_masks']
            self.flags = load_dict['Flags']
    def save_dataset(self, fp):
        save_dict = {'Input_ids': self.input_ids,
                     'Attention_masks': self.attention_masks,
                     'Forecast_predictions': self.forecast_predictions,
                     'Correct_answers': self.correct_answers,
                     'Question_input_ids': self.question_input_ids,
                     'Question_attention_masks': self.question_attention_masks,
                     'Flags': self.flags}
        with open(fp, "wb") as f:
            pickle.dump(save_dict, f)

    def __getitem__(self, index):
        d = {'Input_ids': self.input_ids[index],
             'Attention_masks': self.attention_masks[index],
             'Forecast_predictions': self.forecast_predictions[index],
             'Correct_answers': self.correct_answers[index],
             'Flag': self.flags[index]}
             # 'Question_attention_mask': self.question_attention_masks[index],
             # 'Question_input_ids': self.question_input_ids[index]}
        return d
    def __len__(self):
        return len(self.input_ids)
    def sort_by_hardest_baseline(self):
        daily_dic = Pair()
        total_dic = Pair()
        print(len(self.questions))
        for question in self.questions:
            y_save, maj_total, maj_daily, weighted_daily, weighted_total, l1, l2, l3, l4, l5, l6, l7, f5 = question.build_all_baselines()
            maj_daily/=f5
            weighted_daily/=f5
            maj_total/=f5
            weighted_total/=f5
            max_daily_baseline = max(maj_daily, weighted_daily)
            max_total_baseline = max(maj_total, weighted_total)
            daily_dic.add(max_daily_baseline, question)
            total_dic.add(max_total_baseline, question)
        daily_dic.sort_all()
        total_dic.sort_all()
        length = len(daily_dic.values)
        # for i in range(len(daily_dic.values)):
        #     print(total_dic.keys[i], total_dic.values[i].title)
        # with open("Hardest_Questions/Total_Hardest/q4/total_q4.save", "wb") as f:
        #     pickle.dump(total_dic.values[int(length/4)*3:], f)

class Pair:
    def __init__(self):
        self.keys = []
        self.values = []
    def add(self, key, val):
        self.keys.append(key)
        self.values.append(val)
    def sort_all(self):
        new_keys = []
        new_vals = []
        copy_of_keys = self.keys.copy()
        self.keys.sort()
        for i in range(len(self.keys)):
            new_keys.append(self.keys[i])
            for j in range(len(copy_of_keys)):
                if copy_of_keys[j] == self.keys[i]:
                    if self.values[j] not in new_vals:
                        new_vals.append(self.values[j])
        self.keys = new_keys.copy()
        self.values = new_vals.copy()


class Collate:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.empty_encoding = self.tokenizer("", padding="max_length", truncation=True, add_special_tokens=True)
    def __call__(self, batch):
        max_len = 0
        for item in batch:
            max_len = max(max_len, len(item["Input_ids"]))
        input = []
        attention = []
        predictions = []
        correct_answer = []
        # question_input = []
        # question_attention = []
        for item in batch:
            correct_answer.append(item["Correct_answers"])
            input_ids = item["Input_ids"]
            attention_mask = item["Attention_masks"]
            forecast_predictions = item["Forecast_predictions"]
            # question_input_ids = item['Question_input_ids']
            # question_attention_mask = item['Question_attention_mask']
            while len(input_ids) < max_len:
                input_ids.append(self.empty_encoding["input_ids"])
                attention_mask.append(self.empty_encoding["attention_mask"])
                forecast_predictions.append(-1)
                # question_input_ids.append(self.empty_encoding["input_ids"])
                # question_attention_mask.append(self.empty_encoding["attention_mask"])
            input.append(input_ids)
            attention.append(attention_mask)
            predictions.append(forecast_predictions)
            # question_input.append(question_input_ids)
            # question_attention.append(question_attention_mask)
        correct_answer = torch.tensor(correct_answer, dtype = torch.float)
        input = torch.tensor(input, dtype = torch.long)
        predictions = torch.tensor(predictions, dtype = torch.float)
        correct_answer = torch.tensor(correct_answer, dtype = torch.float)
        attention = torch.tensor(attention, dtype = torch.long)
        # question_input = torch.tensor(question_input, dtype = torch.long)
        # question_attention = torch.tensor(question_attention, dtype = torch.long)
        d = {'Input_ids': input,
             'Attention_masks': attention,
             'Forecast_predictions': predictions,
             'Correct_answers': correct_answer}
             # 'Question_attention_mask': question_attention,
             # 'Question_input_ids': question_input}
        return d
def get_dataloaders(batch_size, daily, question_fp):
    with open(question_fp, "rb") as f:
        question_dict = pickle.load(f)
    train_data = GJO_Dataset(question_dict["Train"])
    test_data = GJO_Dataset(question_dict["Test"])
    val_data = GJO_Dataset(question_dict["Val"])
    if daily:
        fp1 = "Daily/train.save"
        fp2 = "Daily/test.save"
        fp3 = "Daily/val.save"
    else:
        fp1 = "Total/train.save"
        fp2 = "Total/test.save"
        fp3 = "Total/val.save"
    train_data.load_dataset(fp1)
    test_data.load_dataset(fp2)
    val_data.load_dataset(fp3)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=Collate())
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=Collate())
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=Collate())
    return train_loader, test_loader, val_loader
def load(data_dir):
    questions = []
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    for fp in os.listdir(data_dir):
        path = data_dir + fp
        if path.endswith(".json"):
            q = Question(path, bert_tokenizer)
            print(q.id)
            q.df.drop_duplicates(subset=["text"], inplace=True)
            q.df = q.df.sort_values(by='timestamp')
            q.build_day_lists()
            questions.append(q)
    return questions
def do_questions(fp):
    questions = load("./Questions/")
    random.shuffle(questions)
    train_length = int(0.7 * len(questions))
    test_length = int(0.2 * len(questions))
    train_data = questions[:train_length]
    test_data = questions[train_length: train_length + test_length]
    val_data = questions[train_length + test_length:]
    question_dict = {"Train": train_data, "Test": test_data, "Val": val_data}
    with open(fp, "wb") as f:
        pickle.dump(question_dict, f)
def load_questions(fp):
    with open(fp, "rb") as f:
        question_dict = pickle.load(f)
    train_data = question_dict["Train"]
    test_data = question_dict["Test"]
    val_data = question_dict["Val"]
    return train_data, test_data, val_data

def setup(daily, fp, load):
    if not load:
        do_questions(fp)
    train_data, test_data, val_data = load_questions(fp)
    train_data.extend(val_data)
    train_set = GJO_Dataset(train_data)
    test_set = GJO_Dataset(test_data)
    val_set = GJO_Dataset(val_data)

    train_set.do_all_baselines()
    print("####################")
    val_set.do_all_baselines()
    print("####################")
    test_set.do_all_baselines()
    print("#################################")
    train_set.build_dataset(daily)
    test_set.build_dataset(daily)
    val_set.build_dataset(daily)
    if daily:
        fp1 = "Daily/train.save"
        fp2 = "Daily/test.save"
        fp3 = "Daily/val.save"
    else:
        fp1 = "Total/train.save"
        fp2 = "Total/test.save"
        fp3 = "Total/val.save"
    train_set.save_dataset(fp1)
    test_set.save_dataset(fp2)
    val_set.save_dataset(fp3)
def calc_mcnemar(file1, file2):
    with open(file1, "rb") as f:
        baselines_list = pickle.load(f)
    with open(file2, "rb") as f:
        model_list = pickle.load(f)
    both_correct = 0
    onecorrect2incorrect = 0
    oneincorrect2correct = 0
    both_incorrect = 0
    one_correct = 0
    two_correct = 0
    print(sum(baselines_list)/len(baselines_list))
    print(sum(model_list)/len(model_list))
    print(len(model_list), len(baselines_list))
    for i in range(len(baselines_list)):
        if baselines_list[i] == 1 and model_list[i] == 1:
            both_correct+=1
        if baselines_list[i] == 0 and model_list[i] == 0:
            both_incorrect+=1
        if baselines_list[i] == 1 and model_list[i] == 0:
            onecorrect2incorrect+=1
        if baselines_list[i] == 0 and model_list[i] == 1:
            oneincorrect2correct+=1
        if baselines_list[i] == 1:
            one_correct+=1
        if model_list[i] == 1:
            two_correct+=1
    table = [[both_correct, onecorrect2incorrect], [oneincorrect2correct, both_incorrect]]
    result = mcnemar(table, exact = True)
    print(result.pvalue)
if __name__ == "__main__":
    pass