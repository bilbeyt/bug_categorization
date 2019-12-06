import csv
import requests
import progressbar
import pandas as pd

PROJECT_URLS = {
    "lucene": "https://issues.apache.org/jira/rest/api/2/issue/{0}?fields=summary",
    "jackrabbit": "https://issues.apache.org/jira/rest/api/2/issue/{0}?fields=summary",
    "httpclient": "https://issues.apache.org/jira/rest/api/2/issue/{0}?fields=summary"
}

def read_dataset_csv(project_name):
    rows = pd.read_csv(f'datasets/{project_name}.csv')
    return rows

def add_title_to_dataset(project, projects):
    titles = []
    length = len(projects[project]['ID'])
    bar = progressbar.ProgressBar(max_value=length).start()
    for issue_key in bar(projects[project]['ID']):
        url = PROJECT_URLS[project].format(issue_key)
        req = requests.get(url)
        title = req.json()['fields']['summary']
        titles.append(title)
    projects[project]['TITLE'] = titles
    return projects

def write_dataset_to_csv(project, projects):
    data = projects[project]
    data.to_csv(f'datasets/final_{project}.csv', index=False, 
                quoting=csv.QUOTE_ALL)


if __name__ == '__main__':
    datasets = {}
    for project in PROJECT_URLS:
        datasets[project] = read_dataset_csv(project)
    for project in PROJECT_URLS:
        projects = add_title_to_dataset(project, datasets)
    for project in PROJECT_URLS:
        write_dataset_to_csv(project, projects)
