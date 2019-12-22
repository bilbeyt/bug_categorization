import csv
import requests
import progressbar
import pandas as pd


BASE_URL = "https://issues.apache.org/jira/rest/api/2/issue"
PROJECT_URLS = {
    "lucene": "{1}/{0}?fields=summary,description",
    "jackrabbit": "{1}/{0}?fields=summary,description",
    "httpclient": "{1}/{0}?fields=summary,description"
}


def read_dataset_csv(project_name):
    rows = pd.read_csv(f'../datasets/{project_name}.csv')
    return rows


def add_title_to_dataset(project, projects):
    titles = []
    descriptions = []
    length = len(projects[project]['ID'])
    progress_bar = progressbar.ProgressBar(max_value=length).start()
    for issue_key in progress_bar(projects[project]['ID']):
        url = PROJECT_URLS[project].format(issue_key, BASE_URL)
        req = requests.get(url)
        title = req.json()['fields']['summary']
        description = req.json()['fields']['description']
        titles.append(title)
        descriptions.append(description)
    projects[project]['TITLE'] = titles
    projects[project]['DESCRIPTION'] = descriptions
    return projects


def write_dataset_to_csv(project, projects):
    data = projects[project]
    data.to_csv(f'../datasets/final_{project}.csv', index=False,
                quoting=csv.QUOTE_ALL)


if __name__ == '__main__':
    datasets = {}
    for project in PROJECT_URLS:
        datasets[project] = read_dataset_csv(project)
    for project in PROJECT_URLS:
        projects = add_title_to_dataset(project, datasets)
    for project in PROJECT_URLS:
        write_dataset_to_csv(project, projects)
