import string

from collections import Counter
from random import sample

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def calculate_membership_score(freqs):
    result = 1
    for freq in freqs:
        result *= 1 - freq
    return 1 - result


class Dataset:
    STOP_WORDS = set(stopwords.words('english'))

    def __init__(self, source_file, name, used_item_key):
        self.source_file = source_file
        self.name = name
        self.used_item_key = used_item_key
        self.bugs = []
        self.others = []
        self.training = {'bugs': [], 'others': []}
        self.test = {'bugs': [], 'others': []}
        self.vocabulary = {'bugs': [], 'others': []}
        self.confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        self.metrics = {'accuracy': 0, 'f_measure': 0, 'recall': 0, 'precision': 0}
        self.frequencies = {'bugs': {}, 'others': {}}
        self.read_data_from_file()

    def read_data_from_file(self):
        self.data = pd.read_csv(self.source_file)

    def classify_dataset(self):
        for _, row in self.data.iterrows():
            if row.get(self.used_item_key) == 'BUG':
                self.bugs.append(row)
            else:
                self.others.append(row)

    def split_data(self, key):
        data = getattr(self, key)
        train = sample(data, int(len(data)*0.8))
        processed_ids = [x.ID for x in train]
        test = [x for x in data if x.ID not in processed_ids]
        self.training[key] = train
        self.test[key] = test

    def process_titles(self, field_name):
        data = getattr(self, field_name)
        for issue_type, issues in data.items():
            for issue in issues:
                title = issue.TITLE
                title = title.translate(
                    str.maketrans("", "", string.punctuation))
                title = title.lower()
                title = [w for w in word_tokenize(
                    title) if w not in Dataset.STOP_WORDS]
                issue.PROCESSED = title
                self.vocabulary[issue_type] += title

    def prepare_training_frequencies(self):
        bugs = Counter(self.vocabulary['bugs'])
        others = Counter(self.vocabulary['others'])
        all_issues = bugs + others
        for key in bugs:
            term_freq = bugs[key] / all_issues[key]
            self.frequencies['bugs'][key] = term_freq
        for key in others:
            term_freq = others[key] / all_issues[key]
            self.frequencies['others'][key] = term_freq

    def process_test_data(self):
        for _, issues in self.test.items():
            for issue in issues:
                processed_title = issue.PROCESSED
                bug_freqs = [self.frequencies['bugs'].get(
                    w, 0) for w in processed_title]
                others_freqs = [self.frequencies['others'].get(
                    w, 0) for w in processed_title]
                bug_score = calculate_membership_score(bug_freqs)
                others_score = calculate_membership_score(
                    others_freqs)
                if bug_score >= others_score:
                    if issue.get(self.used_item_key) == 'BUG':
                        self.confusion_matrix['TP'] += 1
                    else:
                        self.confusion_matrix['FP'] += 1
                else:
                    if issue.get(self.used_item_key) == 'BUG':
                        self.confusion_matrix['FN'] += 1
                    else:
                        self.confusion_matrix['TN'] += 1

    def calculate_precision(self):
        return (
            self.confusion_matrix['TP'] /
            (self.confusion_matrix['TP'] + self.confusion_matrix['FP'])
        )

    def calculate_recall(self):
        return (
            self.confusion_matrix['TP'] /
            (self.confusion_matrix['TP'] + self.confusion_matrix['FN'])
        )

    def calculate_accuracy(self):
        return (
            (self.confusion_matrix['TP'] + self.confusion_matrix['TN']) /
            sum(self.confusion_matrix.values())
        )

    def calculate_metrics(self):
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        acc = self.calculate_accuracy()
        f_measure = (2 * precision * recall) / (precision + recall)
        self.metrics = {
            'precision': f"{precision:.03f}",
            'recall': f"{recall:.03f}",
            'accuracy': f"{acc:.03f}",
            'f_measure': f"{f_measure:.03f}"
        }

    def report(self):
        print(f"Dataset: {self.name}")
        print(f"TP: {self.confusion_matrix['TP']}")
        print(f"TN: {self.confusion_matrix['TN']}")
        print(f"FP: {self.confusion_matrix['FP']}")
        print(f"FN: {self.confusion_matrix['FN']}")
        print(f"Precision: {self.metrics['precision']}")
        print(f"Recall: {self.metrics['recall']}")
        print(f"Accuracy: {self.metrics['accuracy']}")
        print(f"F-measure: {self.metrics['f_measure']}")

    def run(self):
        self.classify_dataset()
        self.split_data('bugs')
        self.split_data('others')
        self.process_titles('training')
        self.process_titles('test')
        self.prepare_training_frequencies()
        self.process_test_data()
        self.clean_fields()

    def clean_fields(self):
        self.bugs = []
        self.others = []
        self.vocabulary = {'bugs': [], 'others': []}
