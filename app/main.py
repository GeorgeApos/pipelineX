import joblib

from app.utils import import_data, get_file_extension
from data_preprocessing import preprocess_data, resample_data
from model_prediction import recommend_best_algorithm, split_train_test, select_features_target, models


def main():
    print("Starting Machine Learning Pipeline")
    print("Import your data")
    file_path = input("Enter the file path: ")
    data = import_data(file_path, get_file_extension(file_path))
    print("Data imported successfully")
    preprocess_data(data)
    print("Select features and target")
    X, y = select_features_target(data)
    print("Features and target selected successfully")
    print("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print("Data split successfully")
    X_train, y_train = resample_data(X_train, y_train)
    print("The recommended algorithm is: " + recommend_best_algorithm(X_train, y_train).__class__.__name__)
    choice = input("Would you like to use the recommended algorithm? (y/n): ")
    if choice == 'y':
        model = recommend_best_algorithm(X_train, y_train)
    else:
        print("Available algorithms:")
        for name, _ in models:
            print(name)
        selected_model = input("Enter the algorithm name: ")
        for name, model in models:
            if selected_model == name:
                model.fit(X_train, y_train)
                break
        else:
            print("Invalid algorithm choice. Using the recommended algorithm.")
            model = recommend_best_algorithm(X_train, y_train)
    print("Model trained successfully")
    print("Model evaluation:")
    print(f"Training Accuracy: {model.score(X_train, y_train)}")
    print(f"Testing Accuracy: {model.score(X_test, y_test)}")
    print("Model Predictions:")
    print(model.predict(X_test))
    print("Do you want to save the model? (y/n): ")
    choice = input()
    if choice == 'y':
        print("Downloading model...")
        joblib.dump(model, '../models/trained_models/model.pkl')
        print("Model downloaded successfully")


if __name__ == "__main__":
    main()
