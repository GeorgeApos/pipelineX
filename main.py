import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC()),
    ('Naive Bayes', GaussianNB()),
    ('KNN', KNeighborsClassifier()),
    ('Logistic Regression', LogisticRegression()),
    ('Gradient Boosting', GradientBoostingClassifier())
]


def preprocess_data(data):
    print("Do you want to preprocess the data? (y/n): ")
    choice = input()
    if choice == 'y':
        print("Do you want to remove missing values? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Removing missing values...")
            data = data.dropna()
            print("Missing values removed successfully")
        print("Do you want to remove duplicate values? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Removing duplicate values...")
            data = data.drop_duplicates()
            print("Duplicate values removed successfully")
        print("Do you want to encode categorical values? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Which columns do you want to encode? (separated by comma): ")
            columns = input().split(',')

            print("Choose encoding method:")
            print("1. One-Hot Encoding")
            print("2. Label Encoding")
            encoding_choice = int(input("Enter your choice (1 or 2): "))

            if encoding_choice == 1:
                print("One-Hot Encoding categorical values...")
                data = pd.get_dummies(data, columns=columns)
                print("Categorical values encoded successfully")
            elif encoding_choice == 2:
                label_encoder = LabelEncoder()
                for column in columns:
                    data[column] = label_encoder.fit_transform(data[column])
                print("Label Encoding categorical values...")
                print("Categorical values encoded successfully")
            else:
                print("Invalid choice. No encoding applied.")
        print("Do you want to scale numerical values? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Which columns do you want to scale? (separated by comma): ")
            columns = input().split(',')
            print("Scaling numerical values...")
            for column in columns:
                data[column] = data[column] / data[column].max()
            print("Numerical values scaled successfully")
        print("Do you want to normalize numerical values? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Which columns do you want to normalize? (separated by comma): ")
            columns = input().split(',')
            print("Normalizing numerical values...")
            for column in columns:
                data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
            print("Numerical values normalized successfully")
        print("Do you want to bin numerical values? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Which columns do you want to bin? (separated by comma): ")
            columns = input().split(',')
            print("Binning numerical values...")
            for column in columns:
                data[column] = pd.qcut(data[column], 4, labels=False)
            print("Numerical values binned successfully")


def main():
    print("Starting Machine Learning Pipeline")
    print("Import your data")
    file_path = input("Enter the file path: ")
    data = import_data(file_path, "csv")
    print("Data imported successfully")
    preprocess_data(data)
    print("Select features and target")
    X, y = select_features_target(data)
    print("Features and target selected successfully")
    print("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print("Data split successfully")
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
        joblib.dump(model, 'model.pkl')
        print("Model downloaded successfully")


def import_data(file_path, file_type):
    if file_type == 'csv':
        data = pd.read_csv(file_path)
        return data
    elif file_type == 'Excel':
        data = pd.read_excel(file_path)
        return data


def select_features_target(data):
    x_cols = input("Enter the column names for features (separated by comma): ").split(',')
    y_col = input("Enter the column name for the target: ")

    X = data[x_cols].values
    y = data[y_col].values

    return X, y


def split_train_test(X, y):
    test_size = float(input("Enter the test size (as a decimal, e.g., 0.2 for 20%): "))
    random_state = int(input("Enter the random state (an integer for reproducibility): "))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def recommend_best_algorithm(X_train, y_train):
    best_model = None
    best_score = 0

    for name, model in models:
        try:
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            print(f"{name} - Training Accuracy: {score}")
            if score > best_score:
                best_score = score
                best_model = model
        except ValueError as ve:
            print(f"ValueError occurred for {name}: {ve}")
        except TypeError as te:
            print(f"TypeError occurred for {name}: {te}")
        except Exception as e:
            print(f"An error occurred for {name}: {e}")

    return best_model


if __name__ == "__main__":
    main()
