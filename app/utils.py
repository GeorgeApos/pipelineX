import subprocess

import pandas as pd

from sklearn.metrics import recall_score, precision_score, f1_score


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


def evaluate_model(model, X_train, X_test, y_train, y_test, selected_metrics):
    for metric in selected_metrics:
        if metric == 1:
            print(f"Training Accuracy: {model.score(X_train, y_train)}")
            print(f"Testing Accuracy: {model.score(X_test, y_test)}")
        elif metric == 2:
            print(f"Training Recall: {recall_score(y_train, model.predict(X_train))}")
            print(f"Testing Recall: {recall_score(y_test, model.predict(X_test))}")
        elif metric == 3:
            print(f"Training Precision: {precision_score(y_train, model.predict(X_train))}")
            print(f"Testing Precision: {precision_score(y_test, model.predict(X_test))}")
        elif metric == 4:
            print(f"Training F1 Score: {f1_score(y_train, model.predict(X_train))}")
            print(f"Testing F1 Score: {f1_score(y_test, model.predict(X_test))}")


def predict_with_new_data(model):
    if model:
        print("Model Summary:")
        print(model)

        print("Import new data for prediction")
        new_data_path = input("Enter the path to the new data file: ")
        new_data = import_data(new_data_path, get_file_extension(new_data_path))

        print("Making predictions with the model")
        predictions = model.predict(new_data)

        if predictions is not None:
            print("Predictions:")
            print(predictions)
            print("Prediction successfully completed")


def import_data_from_script():
    script_path = input("Enter the path to your Python script for importing data: ")
    try:
        output = subprocess.check_output(["python", script_path], universal_newlines=True)
        file_path = output.strip()
        data = import_data(file_path, get_file_extension(file_path))
        print("Data imported successfully")
        return data
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the script: {e}")
        return None
