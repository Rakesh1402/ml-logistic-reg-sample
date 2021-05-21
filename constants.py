
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# 
# TODO: read from config file
TRAINING_INPUT_PATH = "/home/rakesh/work/ml-logistic-reg-sample/resources/sample-data/adult.data"
TEST_INPUT_PATH = "/home/rakesh/work/ml-logistic-reg-sample/resources/sample-data/adult.test"

TRAINING_CLEAN_INPUT_PATH= "/home/rakesh/work/ml-logistic-reg-sample/resources/clean-sample-data/train.csv"
TEST_CLEAN_INPUT_PATH= "/home/rakesh/work/ml-logistic-reg-sample/resources/clean-sample-data/test.csv"
# constants
DEFAULT_DF_PRINT_ROWS = 3

INPUT_COLUMN_NAMES = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'salary'
]

INPUT_SCHEMA = StructType([
    StructField("age", IntegerType(), True),
    StructField("workclass", StringType(), True),
    StructField("fnlwgt", IntegerType(), True),
    StructField("education", StringType(), True),
    StructField("education_num", IntegerType(), True),
    StructField("marital_status", StringType(), True),
    StructField("occupation", StringType(), True),
    StructField("relationship", StringType(), True),
    StructField("race", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("capital_gain", IntegerType(), True),
    StructField("capital_loss", IntegerType(), True),
    StructField("hours_per_week", IntegerType(), True),
    StructField("native_country", StringType(), True),
    StructField("salary", StringType(), True)
])
