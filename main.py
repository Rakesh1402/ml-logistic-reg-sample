
from pyspark import SparkContext
from pyspark.sql import SparkSession

from RunMode import RunMode
from constants import *
from PreProcessor import PreProcessor
from SparkProcessor import SparkProcessor


def run_scikit_lr():
    print("Going to run SCIKIT based Logistic Regression...")
    PreProcessor.process()


def run_spark_lr():
    print("Going to run Spark Based Logistic Regression...")
    sc = SparkContext('local[*]')
    spark = SparkSession(sc).builder.appName("ml-example").getOrCreate()

    training_df = spark.read.csv(TRAINING_CLEAN_INPUT_PATH, header=False, schema=INPUT_SCHEMA)
    test_df = spark.read.format("csv").option("header", "false").load(TEST_CLEAN_INPUT_PATH).toDF("age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","salary")
    test_df = SparkProcessor.cast_num_fields(test_df)
    #print("bad records in test df")
    #test_df.filter(~fun.col("age").isNull()).show(10, False)
    SparkProcessor.process(training_df, test_df)


def main(run_mode):
    if run_mode == RunMode.SPARK:
        run_spark_lr()
    else:
        run_scikit_lr()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(RunMode.SPARK)

