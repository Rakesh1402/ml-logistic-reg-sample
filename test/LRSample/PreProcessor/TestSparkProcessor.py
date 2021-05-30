import pytest
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from python.LRSample.PreProcessor.SparkProcessor import SparkProcessor


# Arrange
@pytest.fixture(scope="session")
def spark_session():
    print("Going to create spark session")
    spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()
    return spark


# Arrange
@pytest.fixture(scope="session")
@pytest.mark.usefixtures("spark_session")
def test_df(spark_session):
    test_df = spark_session.createDataFrame(
        [
            (25, "Private", 226802, "11th", 7, "Never-married", "Machine-op-inspct", "Own-child", "Black", "Male", 0, 0,
            40, "United-States", "<=50K."),
            (38, "Private", 89814, "HS-grad", 9, "Married-civ-spouse", "Farming-fishing", "Wife", "White", "Female", 0,
            0, 50, "Portugal", "<=50K."),
            (28, "Local-gov", 336951, "Assoc-acdm", 12, "Married-civ-spouse", "Protective-serv", "Husband", "White",
            "Male", 0, 0, 40, "United-States", ">50K."),
            (44, "Private", 160323, "Some-college", 10, "Married-civ-spouse", "Machine-op-inspct", "Husband", "Black",
            "Male", 7688, 0, 40, "United-States", "<=50K.")
        ],
        ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race",
         "sex","capital_gain","capital_loss","hours_per_week","native_country","salary"]
    )
    test_df.show(10, False)
    print("Test df count: {}".format(test_df.count()))
    return test_df


@pytest.mark.usefixtures("spark_session", "test_df")
def test_encode_target_label(spark_session, test_df):
    '''
    Test Cases:
    1. Output dataframe should have one more column label
    2. Output Dataframe's new Label column should have two distinct values
    3. As per test data, salary column ">50K." and "<=50K." should have label 1.0 and 0.0 respectively
    '''
    train_df = test_df
    input_cols_count = len(train_df.columns)
    print("Input df # of columns: {}".format(input_cols_count))
    # Act
    train_out_df, test_out_df = SparkProcessor.encode_target_label(train_df, test_df)
    #print("Encoded Salary column DF:")
    #train_out_df.show(10, False)
    # Assert
    # assert # of columns in output dataframe
    assert len(train_out_df.columns) == input_cols_count + 1

    # assert distinct values of label column
    # for test data distinct values are limited and hence its fine to collect it
    label_col_distinct_values = train_out_df.select("label").distinct().collect()
    print("Distinct values of label: {}".format(len(label_col_distinct_values)))
    assert len(label_col_distinct_values) == 2

    # assert mapping of salary column to label
    grouped_rows = train_out_df.select("salary", "label").distinct().orderBy("salary", "label").collect()
    #mapped_labels = list(map(lambda input_row: (input_row.asDict["salary"], input_row.asDict["label"]), grouped_rows))
    grouped_rows_dict_list = list(map(lambda input_row: get_tuple(input_row), grouped_rows))
    grouped_dict = {item['salary']: item['label'] for item in grouped_rows_dict_list}
    #assert train_out_df.filter(col("salary") == "<=50K.").select("label").distinct().collect()[0].asDict()["label"] == 0.0
    assert grouped_dict[">50K."] == 1.0
    assert grouped_dict["<=50K."] == 0.0


def get_tuple(input_row):
    input_dict = input_row.asDict()
    #return input_dict["salary"], input_dict["label"]
    return input_dict

@pytest.mark.usefixtures("spark_session", "test_df")
def test_encode_categorical_cols(spark_session, test_df):
    print("Executing spark preprocessor test...")
    assert test_df.count() == 4
    train_df = test_df
    train_df, test_df = SparkProcessor.encode_categorical_cols(train_df, test_df, categorical_variables = ['workclass', 'education'])
    print("Encoded output df: {}".format(train_df.count))

    #assert new_df.toPandas().to_dict('list')['new_column'][0] == '70'

