import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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


def main():
    print("Starting Machine Learning Pipeline")
    print("Import your data")
    file_path = input("Enter the file path: ")
    file_type = input("Enter the file type (CSV or Excel): ")
    data = import_data(file_path, file_type)
    print("Data imported successfully")
    # pre-processing
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
    if file_type == 'CSV':
        data = pd.read_csv(file_path)
        return data
    elif file_type == 'Excel':
        data = pd.read_excel(file_path)
        return data


def select_features_target(data):
    print("Columns in your dataset:")
    print(data.columns)

    x_cols = input("Enter the column numbers for features (separated by comma): ").split(',')
    y_col = input("Enter the column number for the target: ")

    x_cols = [int(col) for col in x_cols]
    y_col = int(y_col)

    X = data.iloc[:, x_cols].values
    y = data.iloc[:, y_col].values

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
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)
        print(f"{name} - Training Accuracy: {score}")
        if score > best_score:
            best_score = score
            best_model = model

    return best_model


if __name__ == "__main__":
    main()
