from models import LDADataset, FuzzyDataset


class DatasetCategorizer:
    def __init__(self):
        self.dataset_infos = {
            'lucene': '../datasets/final_lucene.csv',
            'jackrabbit': '../datasets/final_jackrabbit.csv',
            'httpclient': '../datasets/final_httpclient.csv',
        }
        self.batch = 5
        self.datasets = []
        self.used_fields = ['titles', 'descriptions']
        self.compares = {'CLASSIFIED': 'Manuel', 'TYPE': 'JIRA'}

    def initialize_datasets(self, used_key, used_field):
        for d_name, d_source_file in self.dataset_infos.items():
            if used_field == 'titles':
                dataset = FuzzyDataset(d_source_file, d_name, used_key)
            else:
                dataset = LDADataset(d_source_file, d_name, used_key)
            self.datasets.append(dataset)

    def run(self):
        for used_field in self.used_fields:
            print(f"=========={used_field}==========")
            for key, value in self.compares.items():
                self.datasets = []
                self.initialize_datasets(key, used_field)
                print(f"=========={value}==========")
                for dataset in self.datasets:
                    if used_field == 'titles':
                        for _ in range(self.batch):
                            dataset.run()
                    else:
                        dataset.run()
                    dataset.calculate_metrics()
                    dataset.report()
                    print(f"========================")


if __name__ == "__main__":
    categorizer = DatasetCategorizer()
    categorizer.run()
