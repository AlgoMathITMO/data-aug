## Utility class extensions

from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import monotonically_increasing_id, row_number, column

import pickle


class NotFittedError(Exception):
    pass

class NotInitializedError(Exception):
    pass

class RealDataNotPresented(Exception):
    pass


# def create_index(df : DataFrame, col_name : str = 'index'):
#     df = df.withColumn('monotonically_increasing_id', monotonically_increasing_id())
#     window = Window.orderBy(df['monotonically_increasing_id'])
#     df = df.withColumn(col_name, row_number().over(window) - 1).drop('monotonically_increasing_id')

#     return df

def create_index(df : DataFrame, col_name : str = 'index'):
    columns = df.columns

    rdd_df = df.rdd.zipWithIndex()
    df = rdd_df.toDF()

    for c in columns:
        df = df.withColumn(c, df['_1'].getItem(c))

    return df.select(column('_2').alias(col_name), *columns)

# TODO: this shuffles the rows after join
def stack_dataframes(df1 : DataFrame, df2 : DataFrame):
    df1 = create_index(df1, col_name='row_index')
    df2 = create_index(df2, col_name='row_index')

    return df1.join(df2, on='row_index').drop('row_index')





def save(obj : object, filename : str):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load(filename : str):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj
