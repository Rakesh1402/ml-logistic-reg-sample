import numpy as np
from python.LRSample.Common.CommonUtil import *


class PreProcessor:
    def __init__(self):
        print("Creating instance of Preprocessor..")

    @staticmethod
    def process():
        print("Starting preprocessing")
        train_df = pd.read_csv(Constants.TRAINING_INPUT_PATH, names=Constants.INPUT_COLUMN_NAMES)
        test_df = pd.read_csv(Constants.TEST_INPUT_PATH, names=Constants.INPUT_COLUMN_NAMES)
        clean_train_df, clean_test_df = PreProcessor.clean_input_df(train_df, test_df)
        PreProcessor.normalize_df(clean_train_df, clean_test_df)

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
