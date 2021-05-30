from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, IntegerType, StringType


class SparkProcessor:
    def __init__(self):
        print("Creating instance of Spark Preprocessor")

    @staticmethod
    def process(train_df, test_df):
        train_df, test_df = SparkProcessor.encode_categorical_cols(train_df, test_df)
        train_df, test_df = SparkProcessor.assemble(train_df, test_df)
        train_df, test_df = SparkProcessor.encode_target_label(train_df, test_df)
        model = SparkProcessor.fit_model(train_df)
        pred_df = SparkProcessor.predict(model, test_df)
        return train_df, test_df, pred_df

    @staticmethod
    def encode_categorical_cols(train_df, test_df, categorical_variables=None):
        if categorical_variables is None:
            categorical_variables = ['workclass', 'education', 'marital_status', 'occupation',
                                     'relationship', 'race', 'sex', 'native_country']

        indexers = [StringIndexer(inputCol=column, outputCol=column + "-index") for column in categorical_variables]
        encoder = OneHotEncoderEstimator(
            inputCols=[indexer.getOutputCol() for indexer in indexers],
            outputCols=["{0}-encoded".format(indexer.getOutputCol()) for indexer in indexers]
        )
        assembler = VectorAssembler(
            inputCols=encoder.getOutputCols(),
            outputCol="categorical-features"
        )
        pipeline = Pipeline(stages=indexers + [encoder, assembler])
        train_df = pipeline.fit(train_df).transform(train_df)
        test_df = pipeline.fit(test_df).transform(test_df)
        train_df.printSchema()
        train_df.show(3, False)
        return train_df, test_df

    @staticmethod
    def encode_target_label(train_df, test_df):
        indexer = StringIndexer(inputCol='salary', outputCol='label')
        train_df = indexer.fit(train_df).transform(train_df)
        test_df = indexer.fit(test_df).transform(test_df)
        return train_df, test_df

    @staticmethod
    def assemble(train_df, test_df):
        continuous_variables = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        assembler = VectorAssembler(
            inputCols=['categorical-features'] + continuous_variables,
            outputCol='features'
        )
        train_df = assembler.transform(train_df)
        train_df.printSchema()
        train_df.show(3, False)
        test_df = assembler.transform(test_df)
        return train_df, test_df

    @staticmethod
    def cast_num_fields(input_df):
        input_df.printSchema()
        # TODO: Optimize this by selecting instead of with Column
        input_df = input_df.withColumn("age", input_df.age.cast("int"))
        input_df = input_df.withColumn("fnlwgt", input_df.fnlwgt.cast("int"))
        input_df = input_df.withColumn("education_num", input_df.education_num.cast("int"))
        input_df = input_df.withColumn("capital_gain", input_df.capital_gain.cast("int"))
        input_df = input_df.withColumn("capital_loss", input_df.capital_loss.cast("int"))
        input_df = input_df.withColumn("hours_per_week", input_df.hours_per_week.cast("int"))
        #input_df.fillna(0).show(10, False)
        return input_df

    @staticmethod
    def fit_model(train_df):
        lr = LogisticRegression(featuresCol='features', labelCol='label')
        model = lr.fit(train_df)
        return model

    @staticmethod
    def predict(model, test_df):
        pred_df = model.transform(test_df)
        pred_df.select("label", "prediction").show(10, False)
        # .select(pred_df.label.cast("int"), pred_df.prediction.cast("int"))\
        # calculate score of the prediction
        print("Accuracy: {}".format(model.summary.accuracy))
        error_df = pred_df.withColumn("error", pred_df.label.isNotNull() & (pred_df.label != pred_df.prediction))
        error_count = error_df.filter(error_df.error == True).count()
        total_count = test_df.count()
        error_per = error_count * 100.0 / total_count
        print("Total samples: {}, error records: {}, Error percentage: {}, Success rate: {}"
              .format(total_count, error_count, error_per, 100 - error_per))
        error_df.show(10, False)
        return pred_df

