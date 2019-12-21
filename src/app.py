from models import Dataset


class FuzzyCategorize:
    def __init__(self):
        self.dataset_infos = {
            'lucene': '../datasets/final_lucene.csv',
            'jackrabbit': '../datasets/final_jackrabbit.csv',
            'httpclient': '../datasets/final_httpclient.csv',
        }
        self.datasets = []
        self.compares = {'CLASSIFIED': 'Manuel', 'TYPE': 'JIRA'}

    def initialize_datasets(self, used_key):
        for d_name, d_source_file in self.dataset_infos.items():
            dataset = Dataset(d_source_file, d_name, used_key)
            self.datasets.append(dataset)

    def run(self):
        for key, value in self.compares.items():
            self.datasets = []
            self.initialize_datasets(key)
            print(f"=========={value}==========")
            for dataset in self.datasets:
                for _ in range(5):
                    dataset.run()
                dataset.calculate_metrics()
                dataset.report()
                print(f"========================")

if __name__ == "__main__":
    categorizer = FuzzyCategorize()
    categorizer.run()
