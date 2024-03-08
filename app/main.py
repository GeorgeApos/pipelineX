import joblib

from app.utils import import_data, get_file_extension, extract_file_name, evaluate_model, predict_with_new_data, \
    import_data_from_script
from data_preprocessing import preprocess_data, resample_data
from model_prediction import recommend_best_algorithm, split_train_test, select_features_target, models, fit_model, \
    tune_hyperparameters
from sklearn.metrics import recall_score, precision_score, f1_score


def main():
    print("Starting Machine Learning Pipeline")

    while True:
        try:
            print("Import your data")
            choice = input("Do you want to import data from a script? (y/n): ")
            if choice.lower() == 'y':
                data = import_data_from_script()
            else:
                file_path = input("Enter the file path: ")
                data = import_data(file_path, get_file_extension(file_path))
                print("Data imported successfully")

            if data is None:
                continue

            preprocess_data(data)

            print("Select features and target")
            X, y = select_features_target(data)
            print("Features and target selected successfully")

            print("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = split_train_test(X, y)
            print("Data split successfully")

            print("Resampling data")
            X_train, y_train = resample_data(X_train, y_train)

            choice = input("Would you like to use the recommended algorithm? (y/n): ")
            if choice == 'y':
                model = recommend_best_algorithm(X_train, y_train)
                if model:
                    fit_model(model, X_train, y_train)
                else:
                    print("No model recommended.")
            else:
                print("Available algorithms:")
                for name, _ in models:
                    print(name)
                selected_model = input("Enter the algorithm name: ")
                for name, model in models:
                    if selected_model == name:
                        model = model
                        break
                else:
                    print("Invalid algorithm choice. Using the recommended algorithm.")
                    model = recommend_best_algorithm(X_train, y_train)

            print("Do you want to fine-tune the model? (y/n): ")
            choice = input()
            if choice == 'y':
                model = tune_hyperparameters(model, X_train, y_train)
                if model:
                    fit_model(model, X_train, y_train)
                else:
                    print("No model recommended.")

            print("Choose evaluation metrics:")
            print("1. Accuracy")
            print("2. Recall")
            print("3. Precision")
            print("4. F1 Score")
            metrics_choice = input("Enter the numbers of the metrics you want to use (comma-separated): ")
            selected_metrics = [int(x) for x in metrics_choice.split(',')]

            evaluate_model(model, X_train, X_test, y_train, y_test, selected_metrics)



            print("Model Predictions:")
            print(model.predict(X_test))

            choice = input("Do you want to save the model? (y/n): ")
            if choice == 'y':
                print("Downloading model...")
                joblib.dump(model, f"models/trained_models/{extract_file_name(file_path)}.pkl")
                print("Model downloaded successfully")

            execute_model = input("Do you want to execute the model with new data? (y/n): ")
            if execute_model.lower() == 'y':
                predict_with_new_data(model)
            else:
                break

            execute_pipeline = input("Do you want to reexecute the pipeline with new data? (y/n): ")
            if execute_pipeline.lower() != 'y':
                break

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
