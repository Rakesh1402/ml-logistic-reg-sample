
from pyspark import SparkContext
from pyspark.sql import SparkSession

from python.LRSample.Common.RunMode import RunMode
from python.LRSample.Common.Constants import *
from python.LRSample.PreProcessor.PreProcessor import PreProcessor
from python.LRSample.PreProcessor.SparkProcessor import SparkProcessor


class LRMain:
    def __init__(self):
        print("Going to create instance of LRMain")

    @staticmethod
    def run_scikit_lr():
        print("Going to run SCIKIT based Logistic Regression...")
        PreProcessor.process()

    @staticmethod
    def run_spark_lr():
        print("Going to run Spark Based Logistic Regression...")
        sc = SparkContext('local[*]')
        spark = SparkSession(sc).builder.appName("ml-example").getOrCreate()

        training_df = spark.read.csv(Constants.TRAINING_CLEAN_INPUT_PATH, header=False, schema=Constants.INPUT_SCHEMA)
        test_df = spark.read.format("csv").option("header", "false").load(Constants.TEST_CLEAN_INPUT_PATH).toDF("age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","salary")
        test_df = SparkProcessor.cast_num_fields(test_df)
        #print("bad records in test df")
        #test_df.filter(~fun.col("age").isNull()).show(10, False)
        SparkProcessor.process(training_df, test_df)

    @staticmethod
    def run_lr_main(run_mode):
        if run_mode == RunMode.SPARK:
            LRMain.run_spark_lr()
        else:
            LRMain.run_scikit_lr()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    LRMain.run_lr_main(RunMode.SPARK)

