import pandas as pd


def import_data(file_path, file_type):
    if file_type == 'csv':
        data = pd.read_csv(file_path)
        return data
    elif file_type == 'Excel':
        data = pd.read_excel(file_path)
        return data


def extract_file_name(file_path):
    return file_path.split('\\')[-1].split('/')[-1].split('.')[0]


def get_file_extension(file_path):
    return file_path.split('.')[-1]
