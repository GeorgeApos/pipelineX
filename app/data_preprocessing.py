import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler


def resample_data(X_train, y_train):
    print("Do you want to resample the data? (y/n): ")
    choice = input()
    if choice == 'y':
        print("Choose resampling techniques:")
        print("1. Random Oversampling")
        print("2. Random Undersampling")
        print("3. SMOTE")
        print("4. ADASYN")
        print("Enter the numbers of the resampling techniques you want to use (comma-separated): ")

        resampling_choices = input()
        selected_resampling_techniques = [int(x) for x in resampling_choices.split(',')]

        for resampling_choice in selected_resampling_techniques:
            if resampling_choice == 1:
                print("Performing Random Oversampling...")
                ros = RandomOverSampler()
                X_train, y_train = ros.fit_resample(X_train, y_train)
                print("Random Oversampling performed successfully")
            elif resampling_choice == 2:
                print("Performing Random Undersampling...")
                rus = RandomUnderSampler()
                X_train, y_train = rus.fit_resample(X_train, y_train)
                print("Random Undersampling performed successfully")
            elif resampling_choice == 3:
                print("Performing SMOTE...")
                smote = SMOTE()
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print("SMOTE performed successfully")
            elif resampling_choice == 4:
                print("Performing ADASYN...")
                adasyn = ADASYN()
                X_train, y_train = adasyn.fit_resample(X_train, y_train)
                print("ADASYN performed successfully")
            else:
                print(f"Invalid choice: {resampling_choice}. Skipping...")
    return X_train, y_train


def remove_columns(data):
    print("Which columns do you want to remove? (separated by comma): ")
    columns = input().split(',')
    data = data.drop(columns, axis=1)
    print("Columns removed successfully")
    return data


def remove_missing_values(data):
    print("Removing missing values...")
    init_size = len(data.index)
    data = data.dropna()
    data = data.reset_index(drop=True)
    print(f"{init_size - len(data.index)} samples were removed as missing values")
    return data


def remove_duplicate_values(data):
    print("Removing duplicate values...")
    return data.drop_duplicates()


def remove_outliers(data, columns):
    print("Removing outliers...")
    init_size = len(data.index)
    lof = LocalOutlierFactor()
    for column in columns:
        data = data[(lof.fit_predict(data[column].values.reshape(-1, 1)) == 1)]
    print(f"{init_size - len(data.index)} samples were removed as outliers")
    return data


def encode_categorical_values(data, columns, encoding_choice):
    if encoding_choice == 1:
        print("One-Hot Encoding categorical values...")
        return pd.get_dummies(data, columns=columns)
    elif encoding_choice == 2:
        label_encoder = LabelEncoder()
        for column in columns:
            data[column] = label_encoder.fit_transform(data[column])
        print("Label Encoding categorical values...")
        return data
    else:
        print("Invalid choice. No encoding applied.")
        return data


def scale_numerical_values(data, columns):
    print("Scaling numerical values...")
    for column in columns:
        data[column] = data[column] / data[column].max()
    print("Numerical values scaled successfully")
    return data


def normalize_numerical_values(data, columns, min_range=0, max_range=100):
    print("Normalizing numerical values...")
    for column in columns:
        data[column] = (((data[column] - data[column].min()) / (data[column].max() - data[column].min())) *
                        (max_range - min_range) + min_range)
    print("Numerical values normalized successfully")
    return data


def bin_numerical_values(data, columns):
    print("Binning numerical values...")
    for column in columns:
        data[column] = pd.qcut(data[column], 4, labels=False)
    print("Numerical values binned successfully")
    return data


def preprocess_data(data):
    print("Do you want to preprocess the data? (y/n): ")
    choice = input()
    if choice == 'y':
        print("Do you want to remove columns? (y/n): ")
        choice = input()
        if choice == 'y':
            data = remove_columns(data)

        print("Do you want to remove missing values? (y/n): ")
        choice = input()
        if choice == 'y':
            data = remove_missing_values(data)

        print("Do you want to remove duplicate values? (y/n): ")
        choice = input()
        if choice == 'y':
            data = remove_duplicate_values(data)

        print("Do you want to remove outliers (Local Outlier Factor)? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Which columns do you want to remove outliers from? (separated by comma): ")
            columns = input().split(',')
            data = remove_outliers(data, columns)

        print("Do you want to encode categorical values? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Which columns do you want to encode? (separated by comma): ")
            columns = input().split(',')
            print("Choose encoding method:")
            print("1. One-Hot Encoding")
            print("2. Label Encoding")
            encoding_choice = int(input("Enter your choice (1 or 2): "))
            data = encode_categorical_values(data, columns, encoding_choice)

        print("Do you want to scale numerical values? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Which columns do you want to scale? (separated by comma): ")
            columns = input().split(',')
            data = scale_numerical_values(data, columns)

        print("Do you want to normalize numerical values? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Which columns do you want to normalize? (separated by comma): ")
            columns = input().split(',')
            data = normalize_numerical_values(data, columns)

        print("Do you want to bin numerical values? (y/n): ")
        choice = input()
        if choice == 'y':
            print("Which columns do you want to bin? (separated by comma): ")
            columns = input().split(',')
            data = bin_numerical_values(data, columns)

    return data
