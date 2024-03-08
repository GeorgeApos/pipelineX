from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

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


def grid_search_hyperparameters(model, X_train, y_train):

    if model.__class__.__name__ == 'DecisionTreeClassifier':
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    elif model.__class__.__name__ == 'RandomForestClassifier':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    elif model.__class__.__name__ == 'SVC':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3, 4, 5],
            'gamma': ['scale', 'auto']
        }
    elif model.__class__.__name__ == 'GradientBoostingClassifier':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 1],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model.__class__.__name__ == 'KNeighborsClassifier':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    elif model.__class__.__name__ == 'LogisticRegression':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'max_iter': [100, 200, 300]
        }
    else:
        print("No hyperparameters to tune for this model.")

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def tune_hyperparameters(model, X_train, y_train):
    print("Do you want to tune the hyperparameters? (y/n): ")
    choice = input()
    if choice == 'y':
        print("Do you want to tune the hyperparameters manually or use Grid Search? (m/g): ")
        choice = input()

        if choice == 'm':
            print("Manual hyperparameter tuning...")
            if model.__class__.__name__ == 'DecisionTreeClassifier':
                criterion = input("Enter the criterion (gini or entropy): ")
                max_depth = int(input("Enter the maximum depth: "))
                min_samples_split = int(input("Enter the minimum samples split: "))
                min_samples_leaf = int(input("Enter the minimum samples leaf: "))
                max_features = input("Enter the maximum features (auto, sqrt, log2, or a number): ")

                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf, max_features=max_features)
            elif model.__class__.__name__ == 'RandomForestClassifier':
                n_estimators = int(input("Enter the number of estimators: "))
                criterion = input("Enter the criterion (gini or entropy): ")
                max_depth = int(input("Enter the maximum depth: "))
                min_samples_split = int(input("Enter the minimum samples split: "))
                min_samples_leaf = int(input("Enter the minimum samples leaf: "))
                max_features = input("Enter the maximum features (auto, sqrt, log2, or a number): ")

                model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                               max_features=max_features)
            elif model.__class__.__name__ == 'SVC':
                C = float(input("Enter the regularization parameter (C): "))
                kernel = input("Enter the kernel (linear, poly, rbf, or sigmoid): ")
                degree = int(input("Enter the degree for the polynomial kernel: "))
                gamma = input("Enter the kernel coefficient (scale or auto): ")

                model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
            elif model.__class__.__name__ == 'GradientBoostingClassifier':
                n_estimators = int(input("Enter the number of estimators: "))
                learning_rate = float(input("Enter the learning rate: "))
                max_depth = int(input("Enter the maximum depth: "))
                min_samples_split = int(input("Enter the minimum samples split: "))
                min_samples_leaf = int(input("Enter the minimum samples leaf: "))

                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                                   max_depth=max_depth, min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf)
            elif model.__class__.__name__ == 'GaussianNB':
                print("No hyperparameters to tune for Gaussian Naive Bayes.")
            elif model.__class__.__name__ == 'KNeighborsClassifier':
                n_neighbors = int(input("Enter the number of neighbors: "))
                weights = input("Enter the weight function (uniform or distance): ")
                algorithm = input("Enter the algorithm (auto, ball_tree, kd_tree, or brute): ")

                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
            elif model.__class__.__name__ == 'LogisticRegression':
                C = float(input("Enter the regularization parameter (C): "))
                max_iter = int(input("Enter the maximum number of iterations: "))

                model = LogisticRegression(C=C, max_iter=max_iter)
        elif choice == 'g':
            print("Using Grid Search for hyperparameter tuning...")
            model = grid_search_hyperparameters(model, X_train, y_train)
            print("Grid Search hyperparameter tuning completed.")
        else:
            print("No hyperparameter tuning performed")
    return model


def recommend_best_algorithm(X_train, y_train):
    best_model = None
    best_f1_score = 0

    for name, model in models:
        try:

            model.fit(X_train, y_train)
            accuracy = model.score(X_train, y_train)
            recall = recall_score(y_train, model.predict(X_train))
            precision = precision_score(y_train, model.predict(X_train))
            f1 = f1_score(y_train, model.predict(X_train))

            print(f"{name}:")
            print(f"  Accuracy: {accuracy}")
            print(f"  Recall: {recall}")
            print(f"  Precision: {precision}")
            print(f"  F1 Score: {f1}")

            if f1 > best_f1_score:
                best_f1_score = f1
                best_model = model

        except ValueError as ve:
            print(f"ValueError occurred for {name}: {ve}")
        except TypeError as te:
            print(f"TypeError occurred for {name}: {te}")
        except Exception as e:
            print(f"An error occurred for {name}: {e}")

    print("Recommendation based on F1 Score:")
    print(f"Best Model: {best_model.__class__.__name__}")
    print(f"Best F1 Score: {best_f1_score}")

    return best_model


def fit_model(model, X_train, y_train):
    try:
        model.fit(X_train, y_train)
        print("Model trained successfully")
    except Exception as e:
        print(f"An error occurred while training the model: {e}")
