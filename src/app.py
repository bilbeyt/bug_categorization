from collections import defaultdict
import pandas as pd
import string
from random import sample
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from collections import Counter


def calculate_membership_score(freqs):
    result = 1
    for freq in freqs:
        result *= 1 - freq
    return 1 - result


class FuzzyCategorize:
    def __init__(self):
        self.datasets = {
            'lucene': '../datasets/final_lucene.csv',
            'jackrabbit': '../datasets/final_jackrabbit.csv',
            'httpclient': '../datasets/final_httpclient.csv',
        }
        self.stop_words = set(stopwords.words('english')) 
        self.data = defaultdict(dict)
        self.frequencies = {
            'bug': {},
            'others': {}
        }
        self.results = {
            'lucene': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
            'jackrabbit': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
            'httpclient': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        }
        self.metrics = {}
        self.bug_terms = []
        self.other_terms = []
        self.read_datasets()
        self.classify_datasets()
        self.create_training_and_test_data()
        self.process_titles()
        self.prepare_training_frequencies()
        self.process_test_data()
        self.calculate_metrics()

    
    def read_datasets(self):
        for dataset_name, filename in self.datasets.items():
            self.data[dataset_name] = pd.read_csv(filename)

    def classify_datasets(self):
        for dataset_name, data in self.data.items():
            self.data[dataset_name] = defaultdict(list)
            for _, row in data.iterrows():
                if row.CLASSIFIED == 'BUG':
                    self.data[dataset_name]['bug'].append(row)
                else:
                    self.data[dataset_name]['others'].append(row)

    def create_training_and_test_data(self):
        for dataset_name, categorized_data in self.data.items():
            for d_type, d_array in categorized_data.items():
                self.data[dataset_name][d_type] = defaultdict(list)
                train = sample(d_array, int(len(d_array)*0.8))
                processed_ids = [x.ID for x in train]
                test = [x for x in d_array if x.ID not in processed_ids]
                self.data[dataset_name][d_type]['train'] = train
                self.data[dataset_name][d_type]['test'] = test
    
    def process_titles(self):
        for _, categorized_data in self.data.items():
            for _, splitted_data in categorized_data.items():
                for _, d_array in splitted_data.items():
                    for item in d_array:
                        title = item.TITLE
                        title = title.translate(str.maketrans("", "", string.punctuation))
                        title = title.lower()
                        title = [w for w in word_tokenize(title) if w not in self.stop_words]
                        item.PROCESSED = title

    def prepare_training_frequencies(self):
        bugs = []
        others = []
        for _, categorized_data in self.data.items():
            for _, splitted_data in categorized_data.items():
                for key, d_array in splitted_data.items():
                    if key == 'train':
                        for item in d_array:
                            processed_title = item.PROCESSED
                            if item.CLASSIFIED == 'BUG':
                                bugs += processed_title
                            else:
                                others += processed_title
        bugs = Counter(bugs)
        others = Counter(others)
        all_issues = bugs + others
        for key in bugs:
            tf = bugs[key] / all_issues[key]
            self.frequencies['bug'][key] = tf
        for key in others:
            tf = others[key] / all_issues[key]
            self.frequencies['others'][key] = tf

    def process_test_data(self):
        for dataset, categorized_data in self.data.items():
            for _, splitted_data in categorized_data.items():
                for key, d_array in splitted_data.items():
                    if key == 'test':
                        for item in d_array:
                            processed_title = item.PROCESSED
                            bug_freqs = [self.frequencies['bug'].get(w, 0) for w in processed_title]
                            others_freqs = [self.frequencies['others'].get(w, 0) for w in processed_title]
                            bug_score = calculate_membership_score(bug_freqs)
                            others_score = calculate_membership_score(others_freqs)
                            if bug_score >= others_score:
                                if item.TYPE == 'BUG':
                                    self.results[dataset]['TP'] += 1
                                else:
                                    self.results[dataset]['FP'] += 1
                            else:
                                if item.TYPE == 'BUG':
                                    self.results[dataset]['FN'] += 1
                                else:
                                    self.results[dataset]['TN'] += 1
        
    def calculate_metrics(self):
        for dataset, data in self.results.items():
            precision = data['TP'] / (data['TP'] + data['FP'])
            recall = data['TP'] / (data['TP'] + data['FN'])
            acc = (data['TP'] + data['TN']) / sum(data.values())
            f_measure = (2 * precision * recall) / (precision + recall)
            self.metrics[dataset] = {
                'precision': f"{precision:.03f}",
                'recall': f"{recall:.03f}",
                'accuracy': f"{acc:.03f}",
                'f_measure': f"{f_measure:.03f}"
            }
        print(self.metrics)


if __name__ == "__main__":
    categorizer = FuzzyCategorize()