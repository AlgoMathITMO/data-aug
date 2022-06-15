import pandas as pd
from pyspark.sql import SparkSession

from replay.session_handler import State


def pandas_to_spark(df : pd.DataFrame, spark_session : SparkSession = None):
    """
    Converts pandas DataFrame to spark DataFrame

    :param df: DataFrame to convert
    :type df: pd.DataFrame
    :param spark_session: Spark session to use, defaults to None
    :type spark_session: SparkSession, optional
    :return: data converted to spark DataFrame
    :rtype: pyspark.sql.DataFrame
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError('df must be an instance of pd.DataFrame')

    if len(df) == 0:
        raise ValueError('Dataframe is empty')

    if spark_session is not None:
        spark = spark_session
    else:
        # spark = State().session
        spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

    return spark.createDataFrame(df)
