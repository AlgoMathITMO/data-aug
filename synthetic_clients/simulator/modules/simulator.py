from abc import ABC
from typing import Callable, Dict, Iterable, Tuple

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import lit, rand

from simulator.utils import pandas_to_spark, create_index, stack_dataframes
from simulator.utils import NotInitializedError

from simulator.modules import Generator


class Simulator(ABC):
    def __init__(
        self,
        user_generators : Dict[str, Generator],
        item_generator : Generator,
        spark_session : SparkSession = None
    ):
        """
        Creates simulator instanse with user and item generators

        :param user_generators: Defines array of generators to be
            used to generate users. Use a key to address certain
            generator in the init() and sample_users()
        :type user_generators: Dict[str, Generator]
        :param item_generator: Instanse of item generator
        :type item_generator: Generator
        :param spark_session: spark session to use, defaults to None
        :type spark_session: SparkSession, optional
        """

        self.user_generators = user_generators
        self.item_generator = item_generator
        self._initialized = False

        self._generators_decode = {k : i + 1 for i, k in enumerate(user_generators)}

        self.SYNTH_COLUMN_NAME = '__is_synth'
        self.CLUSTER_COLUMN_NAME = '__cluster'

        if spark_session is not None:
            self.spark = spark_session
        else:
            self.spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

    def init(
        self,
        num_users : Dict[str, int],
        num_items : int,
        user_key_col : str = 'user_id',
        item_key_col : str = 'item_id',
        user_df : DataFrame = None,
        item_df : DataFrame = None,
        history_df : DataFrame = None
    ):
        """
        Initialize simulator with synthetic users and synthetic items pool.
        Optionally you may pass the real users and items datasets with
        history log, to be able to run simulation on real dataset or either
        use both real and synthetic

        :param num_users: Number of synthetic users to generate from each of
            the generator passed through constructor
        :type num_users: Dict[str, int]
        :param num_items: Number of synthetic items to generate
        :type num_items: int
        :param user_key_col: column name for user identifier, defaults
            to 'user_id'
        :type user_key_col: str
        :param item_key_col: column name for item identifier, defaults
            to 'item_id'
        :type item_key_col: str
        :param user_df: Dataset with real users and their attributes. Pass
            it, if you want to do simulation on real dataset too. Use the
            'user_id' column as a identifyer to each user. Defaults to None
        :type user_df: DataFrame, optional
        :param item_df: Dataset with real items and their attributes. Pass
            it, if you want to do simulation on real dataset too. Use the
            'item_id' column as a identifyer to each item. Defaults to None
        :type item_df: DataFrame, optional
        :param history_df: Log dataframe. Must contain at least user_id,
            item_id columns. Defaults to None
        :type history_df: DataFrame, optional
        """        

        if (sum(num_users.values()) <= 0 and user_df is None) or\
           (num_items <= 0 and item_df is None):

            raise ValueError('Number of users/items in pool cannot be 0')


        if user_df is not None and user_key_col not in user_df.columns:
                raise ValueError(
                    f'There is no column {user_key_col} in real '\
                     'users dataset'
                )

        if item_df is not None and item_key_col not in item_df.columns:
                raise ValueError(
                    f'There is no column {item_key_col} in real '\
                     'items dataset'
                )

        if history_df is not None and\
                (user_key_col not in history_df.columns or\
                 item_key_col not in history_df.columns):
            raise ValueError(
                f'Some of columns [{item_key_col}, {user_key_col}] '\
                 'are not presented in history dataset'
            )
        
        self.user_key_col = user_key_col
        self.item_key_col = item_key_col

        self.user_df = None
        self.item_df = None

        ## zero cluster for real data
        if user_df is not None:
            self.user_df = user_df.withColumn(self.SYNTH_COLUMN_NAME, lit(0))\
                                  .withColumn(self.CLUSTER_COLUMN_NAME, lit(0))
        if item_df is not None:
            self.item_df = item_df.withColumn(self.SYNTH_COLUMN_NAME, lit(0))

        ## concat dataframes from each of the cluster
        for i, (gen_name, num) in enumerate(num_users.items()):
            if num <= 0:
                continue

            synth_users_df = pandas_to_spark(
                self.user_generators[gen_name].generate(num),
                self.spark
            )
            synth_users_df = create_index(synth_users_df, col_name=user_key_col)
            user_next_id = self.user_df.agg({user_key_col : 'max'})\
                .collect()[0][0] + 1\
                if self.user_df is not None else 0
            synth_users_df = synth_users_df.withColumn(
                user_key_col,
                synth_users_df[user_key_col] + user_next_id
            )
            synth_users_df = synth_users_df.withColumn(
                self.SYNTH_COLUMN_NAME, lit(1)
            )
            synth_users_df = synth_users_df.withColumn(
                self.CLUSTER_COLUMN_NAME, lit(i + 1)
            )

            if self.user_df is not None:
                self.user_df = self.user_df.unionByName(synth_users_df)
            else:
                self.user_df = synth_users_df

        self._num_real_users = self.user_df.filter(
            self.user_df[self.SYNTH_COLUMN_NAME] == 0).count()
        self._num_synth_users = {
            key : self.user_df.filter(
                self.user_df[self.CLUSTER_COLUMN_NAME] == self._generators_decode[key]
            ).count() for key in self.user_generators
        }

        ## cenerate synthetic items 
        synth_items_df = None
        if num_items > 0:
            synth_items_df = pandas_to_spark(
                self.item_generator.generate(num_items),
                self.spark
            )
            synth_items_df = create_index(synth_items_df, col_name=item_key_col)
            item_next_id = self.item_df.agg({item_key_col : 'max'})\
                .collect()[0][0] + 1\
                if self.item_df is not None else 0
            synth_items_df = synth_items_df.withColumn(
                item_key_col,
                synth_items_df[item_key_col] + item_next_id
            )
            synth_items_df = synth_items_df.withColumn(
                self.SYNTH_COLUMN_NAME, lit(1)
            )

            if self.item_df is not None:
                self.item_df = self.item_df.unionByName(synth_items_df)
            else:
                self.item_df = synth_items_df

        self._num_real_items = self.item_df.filter(
            self.item_df[self.SYNTH_COLUMN_NAME] == 0).count()
        self._num_synth_items = self.item_df.filter(
            self.item_df[self.SYNTH_COLUMN_NAME] == 1).count()

        ## create empty history dataframe if no history was provided
        self.history_df = history_df
        if self.history_df is None:
            self.history_df = self.spark.createDataFrame(
                data=self.spark.sparkContext.emptyRDD(),
                schema=StructType([
                    StructField('user_idx', IntegerType(), True),
                    StructField('item_idx', IntegerType(), True),
                    StructField('relevance', DoubleType(), True),
                    StructField('timestamp', IntegerType(), True)
                ])
            )

        self._initialized = True



    def sample_users(
        self,
        num_synth_users : Dict[str, int],
        num_real_users : int = 0
    ) -> DataFrame:
        """
        Samples num_synth_users+num_real_users from users pool

        :param num_synth_users: Number of synthetic users to sample
            from each of the respective generator passed through
            constructor Put -1 for a generator to use all of the
            available users from it
        :type num_synth_users: Dict[str, int]
        :param num_real_users: Number of real users to sample.
            Pass -1 if all of the real users are needed,
            defaults to 0
        :type num_real_users: int, optional
        :return: spark DataFrame of sampled users with their attributes
        :rtype: DataFrame
        """

        ## TODO: Replace orderBy(rand()) with sample

        if not self._initialized:
            raise NotInitializedError('You must call init() first')

        if num_real_users > self._num_real_users:
            raise ValueError('Number of requested real users is '\
                             'greater than presented in the pool')
        
        if num_real_users == -1:
            num_real_users = self._num_real_users
        
        for k, v in num_synth_users.items():
            if v > self._num_synth_users[k]:
                raise ValueError(f'Number of requested synthetic users '\
                                  'from generator {k} is greater than '\
                                  'presented in the pool')
            if v == -1:
                num_synth_users[k] = self._num_synth_users[k]

        ## shuffle users for random sampling
        shuffled_df = self.user_df.orderBy(rand())

        result_df = shuffled_df.filter(shuffled_df[self.SYNTH_COLUMN_NAME] == 0)\
                            .limit(num_real_users)

        for key, num in num_synth_users.items():
            result_df = result_df.unionByName(
                shuffled_df.filter(
                    shuffled_df[self.CLUSTER_COLUMN_NAME] == self._generators_decode[key]
                ).limit(num)
            )

        ## shuffle users
        result_df.orderBy(rand())

        return result_df.drop(self.SYNTH_COLUMN_NAME, self.CLUSTER_COLUMN_NAME)

    def _random_log(
        self,
        user_idx_df : DataFrame,
        item_idx_df : DataFrame,
        response_func : Callable[[DataFrame, DataFrame], Iterable]
    ):
        ## TODO: Replace orderBy(rand()) with sample

        ## get three random items
        random_items = item_idx_df.orderBy(rand()).limit(3)

        action_models = {
            'relevance' : response_func
        }

        history = user_idx_df.crossJoin(random_items)
        history = self.sample_responses(
            recommendations_df=history,
            action_models=action_models
        )
        history = history.withColumn('timestamp', lit(0))

        return history

    def get_train_log(
        self,
        user_df : DataFrame,
        response_func : Callable[[DataFrame, DataFrame], Iterable] = None,
        use_synth_items : bool = True,
        use_real_items : bool = False
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Returns the users, items and log for a given users subset
        to train the recommendation algorithm. If no history was
        provided returns random log

        :param user_df: Users subset dataframe to create train data
        :type user_df: DataFrame
        :param response_func: Function for evaluating response to
            create random log. Should use the following signature:
            (DataFrame, DataFrame), where left/right dataframe
            represents the left/right part of user-item pairs matrix,
            defaults to None
        :type response_func: Callable[[DataFrame, DataFrame], Iterable]
        :param use_synth_items: Whether to use the synthetic items or
            not, defaults to True
        :type use_synth_items: bool, optional
        :param use_real_items: Whether to use the real items or not,
            defaults to True
        :type use_real_items: bool, optional
        :return: Tuple of Log, users, items dataframes
        :rtype: Tuple[DataFrame, DataFrame, DataFrame]
        """

        if user_df.count() == 0:
            raise ValueError('User dataframe is empty')

        if not use_synth_items and not use_real_items:
            raise ValueError(
                'Simulator should use either '\
                'synthetic/real data or both'
            )

        items_cond = None
        if use_synth_items and not use_real_items:
            items_cond = self.item_df[self.SYNTH_COLUMN_NAME] == 1
        elif not use_synth_items and use_real_items:
            items_cond = self.item_df[self.SYNTH_COLUMN_NAME] == 0
        else:
            items_cond = (self.item_df[self.SYNTH_COLUMN_NAME] == 1)\
                       | (self.item_df[self.SYNTH_COLUMN_NAME] == 0)

        items_subset = self.item_df.filter(items_cond)\
                                   .drop(self.SYNTH_COLUMN_NAME)

        log = None
        if self.history_df.count() == 0:
            log = self._random_log(
                user_df.select(self.user_key_col),
                items_subset.select(self.item_key_col),
                response_func
            )
        else:
            history_subset = self.history_df.join(
                user_df,
                self.history_df[self.user_key_col] == user_df[self.user_key_col],
                'leftsemi'
            )
            history_subset = history_subset.join(
                items_subset,
                history_subset[self.item_key_col] == items_subset[self.item_key_col],
                'leftsemi'
            )

            if history_subset.count() <= 0:
                log = self._random_log(
                    user_df.select(self.user_key_col),
                    items_subset.select(self.item_key_col),
                    response_func
                )
            else:
                log = history_subset

        return (
            log,
            user_df,
            items_subset
        )

    def get_user_items(
        self,
        user_df : DataFrame,
        num_synth_items : int = -1,
        num_real_items : int = 0
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Creates candidates to pass to the recommendation algorithm based
        on the provided users

        :param user_df: Users dataframe with features and identifiers. This
            will be returned in a (log, users, items) tuple as is
        :type user_df: DataFrame
        :param num_synth_items: Number of synthetic items to use,
            defaults to -1
        :type num_synth_items: int, optional
        :param num_real_items: Number of real items to use, defaults
            to 0
        :type num_real_items: int, optional
        :return: Tuple of Log, users, items dataframes which will be used
            by recommendation algorithm
        :rtype: Tuple[DataFrame, DataFrame, DataFrame]
        """

        if user_df.count() == 0:
            raise ValueError('User dataframe is empty')

        if num_synth_items == -1:
            num_synth_items = self._num_synth_items
        if num_real_items == -1:
            num_real_items = self._num_real_items

        if num_synth_items <= 0 and num_real_items <= 0:
            raise ValueError(
                'Simulator should use either '\
                'synthetic/real data or both'
            )

        shuffled_df = self.item_df.orderBy(rand())

        items_subset = shuffled_df\
            .filter(shuffled_df[self.SYNTH_COLUMN_NAME] == 1)\
            .limit(num_synth_items)\
            .unionByName(
                shuffled_df\
                    .filter(shuffled_df[self.SYNTH_COLUMN_NAME] == 0)\
                    .limit(num_real_items)
            )\
            .drop(self.SYNTH_COLUMN_NAME)

        return (
            self.history_df,
            user_df,
            items_subset
        )

    def sample_responses(
        self,
        recommendations_df : DataFrame,
        action_models : Dict[str, Callable[[DataFrame, DataFrame], Iterable]],
        save_history : bool = False
    ) -> DataFrame:
        """
        Simulates the actions users took on their recommended items

        :param recommendations_df: Dataframe with recommendations.
            Must contain user's and item's identifier columns to
            define recommended user-item pairs. Other columns will
            be ignored
        :type recommendations_df: DataFrame
        :param action_models: Dictionary of actions that should be
            evaluated for each user-item pair. The key will be used
            to name the action and the value is a function, which
            should use the following signature: (DataFrame, DataFrame),
            where left/right dataframe represents the left/right part
            of user-item pairs matrix
        :type action_models: Dict[str, Callable[[DataFrame, DataFrame], Iterable]]
        :param save_history: Whether to append the actions to the log.
            Will work only if 'rated' action was provided, which
            defines the excistence of a response, defaults to False
        :type save_history: bool, optional
        :return: DataFrame with user-item pairs and the respective actions
        :rtype: DataFrame
        """

        user_matrix = recommendations_df.select(self.user_key_col)\
                        .join(self.user_df, self.user_key_col)\
                        .drop(
                            self.user_key_col,
                            self.SYNTH_COLUMN_NAME,
                            self.CLUSTER_COLUMN_NAME
                        )
        item_matrix = recommendations_df.select(self.item_key_col)\
                        .join(self.item_df, self.item_key_col)\
                        .drop(self.item_key_col, self.SYNTH_COLUMN_NAME)

        actions = {}
        for key, func in action_models.items():
            actions[key] = func(
                user_matrix,
                item_matrix
            )

        actions_df = pandas_to_spark(
            pd.DataFrame(actions),
            self.spark
        )

        pairs_df = recommendations_df\
            .select(self.user_key_col, self.item_key_col)

        result = stack_dataframes(pairs_df, actions_df)

        # TODO: avoid columns mismatch
        if save_history:
            self.history_df = self.history_df.unionByName(
                result.filter(result['rated'] == 1)\
                      .select(self.user_key_col, self.item_key_col, 'relevance')\
                      .withColumn('timestamp', lit(0))
            )

        return result
