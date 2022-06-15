from abc import ABC

from sdv.tabular import CopulaGAN, CTGAN, GaussianCopula, TVAE
import pandas as pd

from simulator.utils import NotFittedError


class Generator(ABC):
    def __init__(
        self
    ):
        raise NotImplementedError()

    def fit(
        data : pd.DataFrame
    ):
        raise NotImplementedError()

    def generate(
        self,
        num_samples : int
    ) -> pd.DataFrame:
        raise NotImplementedError()


class SDVGenerator(Generator):
    def __init__(
        self,
        model='copulagan'
    ):
        """
        Initializes synthetic data generator based on SDV library

        :param model: Model name for table data generation. Options are:
            ['copulagan', 'ctgan', 'gaussiancopula', 'tvae'], defaults
            to 'copulagan'
        :type model: str, optional
        """

        _sdv_model_dict = {
            'copulagan' : CopulaGAN,
            'ctgan' : CTGAN,
            'gaussiancopula' : GaussianCopula,
            'tvae' : TVAE
        }

        self._model = _sdv_model_dict[model]()
        self._fit_called = False

    def fit(
        self,
        data : pd.DataFrame
    ):
        """
        Fits the generator from presented data

        :param data: Data to learn generator from. The data should
            not include any indentifiers, but only features
        :type data: pd.DataFrame
        """

        self._model.fit(data)
        self._fit_called = True

    def generate(
        self,
        num_samples : int
    ) -> pd.DataFrame:
        """
        Generate num_samples samples from learned data distribution

        :param num_samples: Number of samples to generate
        :type num_samples: int
        :return: Dataframe with generated samples
        :rtype: pd.DataFrame
        """

        if not self._fit_called:
            raise NotFittedError('You must call fit first')

        if num_samples < 1:
            raise ValueError('num_samples must be positive value')

        return self._model.sample(num_samples)
