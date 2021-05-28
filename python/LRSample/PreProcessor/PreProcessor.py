import numpy as np
import pandas as pd

from python.LRSample.Common.CommonUtil import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class PreProcessor:
    def __init__(self):
        print("Creating instance of Preprocessor..")

    @staticmethod
    def process():
        print("Starting preprocessing")
        train_df = pd.read_csv(Constants.TRAINING_INPUT_PATH, names=Constants.INPUT_COLUMN_NAMES)
        test_df = pd.read_csv(Constants.TEST_INPUT_PATH, names=Constants.INPUT_COLUMN_NAMES)
        clean_train_df, clean_test_df = PreProcessor.clean_input_df(train_df, test_df)
        X_train, Y_train, X_test, Y_test = PreProcessor.normalize_df(clean_train_df, clean_test_df)
        PreProcessor.predict(X_train, Y_train, X_test, Y_test)

    @staticmethod
    def analyze_input_df(train_df, test_df):
        print("Training data set info:")
        train_df.info()
        CommonUtil.print_df(train_df)
        print("\n\nTesting data set info:")
        test_df.info()
        CommonUtil.print_df(test_df)
        print("\n\n Training data null check:")
        print(train_df.isnull().sum())
        print("\n\n Data types:")
        print(train_df.dtypes.value_counts())
        print("\n\n Categorical values:")
        print(train_df.select_dtypes('object').apply(pd.Series.nunique, axis=0))

    @staticmethod
    def clean_input_df(train_df, test_df):
        train_df = train_df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
        train_df_cp = train_df.copy()
        train_df_cp = train_df_cp.loc[train_df_cp['native_country'] != 'Holand-Netherlands']
        test_df = test_df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
        CommonUtil.print_df(train_df, name="clean_train_df")
        CommonUtil.print_df(test_df, name="clean_test_df")
        PreProcessor.write_df(train_df_cp, test_df)
        return train_df_cp, test_df

    @staticmethod
    def normalize_df(train_df, test_df):
        PreProcessor.analyze_input_df(train_df, test_df)
        train_df = PreProcessor.impute_salary(train_df)
        test_df = PreProcessor.impute_salary(test_df)
        train_df = PreProcessor.one_hot_encoding(train_df)
        test_df = PreProcessor.one_hot_encoding(test_df)
        print('Training Features shape: ', train_df.shape)
        print('Testing Features shape: ', test_df.shape)
        train_df, test_df = PreProcessor.align_training_test_sets(train_df, test_df)
        X_train, Y_train = PreProcessor.split_variables(train_df, 'salary')
        X_test, Y_test = PreProcessor.split_variables(test_df, 'salary')
        X_train, X_test = PreProcessor.scale_features(X_train, X_test)
        return X_train, Y_train, X_test, Y_test

    @staticmethod
    def predict(X_train, Y_train, X_test, Y_test):
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)
        lr_pred = lr.predict(X_test)
        score = accuracy_score(Y_test, lr_pred)
        print("Accuracy score: {}".format(score))
        PreProcessor.plot_predicted(Y_test, lr_pred)

    @staticmethod
    def plot_predicted(y, predicted):
        CommonUtil.print_df(y, 10, "y")
        predicted_df = pd.DataFrame(predicted, columns=['predicted'])
        CommonUtil.print_df(predicted_df, 10, "predicted")
        combined_df = pd.concat([y, pd.Series(predicted)], axis=1, keys = ['salary', 'predicted_salary'])
        error_df = combined_df['salary'] != combined_df['predicted_salary']
        error_count = sum(error_df)
        error_per = sum(error_df)*100.0/y.shape[0]
        print("Error: {}, total: {}, Error %: {}, Success %: {}".format(error_count, y.shape, error_per, 100 - error_per))
        CommonUtil.print_df(combined_df, 10, "combined_df")
        #y.head(100).plot()
        #predicted_df.head(100).plot()
        #plt.show()
        y_index = y.index.tolist()
        plt.plot(y_index[0:1000], y.tolist()[0:1000], label="actual")
        plt.plot(y_index[0:1000], predicted.tolist()[0:1000], label="predicted")
#        plt.plot(y_index, y.tolist(), label="actual")
#        plt.plot(y_index, predicted.tolist(), label="predicted")
        plt.legend()
        plt.show()


    @staticmethod
    def write_df(train_df, test_df):
        train_df.to_csv('/home/rakesh/work/ml-logistic-reg-sample/resources/clean-sample-data/train.csv', index=False, header=False)
        test_df.to_csv('/home/rakesh/work/ml-logistic-reg-sample/resources/clean-sample-data/test.csv', index=False, header=False)

    @staticmethod
    def impute_salary(input_df):
        print(input_df['salary'].unique())
        input_df['salary'] = input_df['salary'].fillna('<=50K')
        match_criteria = [input_df['salary'].str.contains('<=50K'), input_df['salary'].str.contains('>50K')]
        input_df['salary'] = np.select(match_criteria, [0, 1], default=1)
        CommonUtil.print_df(input_df, 3)
        return input_df

    @staticmethod
    def one_hot_encoding(input_df):
        CommonUtil.print_df(input_df, name= "encoding_input_df")
        encoded_df = pd.get_dummies(input_df)
        CommonUtil.print_df(encoded_df, name="encoding_output_df")
        return encoded_df

    @staticmethod
    def align_training_test_sets(train_df, test_df):
        # Align the training and testing data to keep only common columns
        print("Columns in train df: {}".format(",".join(train_df.columns.sort_values())))
        print("Columns in test df: {}".format(",".join(test_df.columns.sort_values())))
        train_df, test_df = train_df.align(test_df, join='inner', axis=1)
        print('Training Features shape: ', train_df.shape)
        print('Testing Features shape: ', test_df.shape)
        return train_df, test_df

    @staticmethod
    def split_variables(input_df, y_column_name):
        X_input = input_df.drop(y_column_name, axis=1)
        Y_input = input_df[y_column_name]
        return X_input, Y_input

    @staticmethod
    def scale_features(X_train, X_test):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test
