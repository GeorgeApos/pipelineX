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
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('Logistic Regression', LogisticRegression(max_iter=1000))
]


def select_features_target(data):
    x_cols = input("Enter the column names for features (separated by comma): ").split(',')
    y_col = input("Enter the column name for the target: ")

    X = data[x_cols].values
    y = data[y_col].values

    return X, y


def split_train_test(X, y):
    test_size = float(input("Enter the test size (as a decimal, e.g., 0.2 for 20%): "))
    random_state = int(input("Enter the random state (an integer for reproducibility): "))
    stratify = input("Do you want to stratify the data? (y/n): ")
    if stratify == 'y':
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=stratify)

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
