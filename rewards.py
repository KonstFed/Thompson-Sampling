from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm


class AbstractReward(ABC):
    """Abstract reward"""

    @abstractmethod
    def get_reward(self, action):
        pass


class BernouliBanditReward(AbstractReward):
    """Bernouli reward for multi arm bandit problem"""

    def __init__(self, thetas: list[float]):
        """
        Args:
            thetas (list[float]): probabilities of success for every arm
        """
        self.thetas = thetas

    def get_reward(self, action: int) -> int:
        """get reward based on which arm is pulled

        Args:
            action (int): which arm is pulled

        Returns:
            int: reward, 0 or 1 value
        """
        return int(np.random.uniform() < self.thetas[action])


class GaussianReward(AbstractReward):
    """Gaussian reward for multi arm bandit problem"""

    def __init__(self, means: list[float], stds: list[float]) -> None:
        """For every arm should provide their normal and standard deviation

        Args:
            means (list[float]): list of means
            stds (list[float]): list of standard deviations
        """
        self.means = means
        self.stds = stds
        super().__init__()

    def get_reward(self, action: int):
        """Reward is sampled from normal distribution from arm's mean and standard deviation

        Args:
            action (int): which arm is pulled

        Returns:
            float: reward
        """
        return norm.ppf(np.random.uniform(), self.means[action], self.stds[action])


class ProductAssortmentReward(AbstractReward):
    """Reward for product assortment problem, see more https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf, page 51"""

    def __init__(self, profits: list[float], demand_std: float, thetas=None) -> None:
        """

        Args:
            profits (list[float]): list of profit for every product.

            demand_std (float): product demand standard deviation

            thetas (np.ndarray, optional): square matrix of how different items influence each others demand. Defaults to None. If not passed, will be randomly generate, see more in `generate_theta` method

        Raises:
            ValueError: thetas is not squared

            ValueError: number of profits should be equal to size of thetas matrix
        """
        self.n_products = len(profits)
        self.profits = profits
        self.demand_std = demand_std
        if thetas is None:
            self.thetas = self.generate_theta()
        elif thetas.shape[0] != thetas.shape[1]:
            raise ValueError("Thetas matrix should be squared")
        elif len(profits) != thetas.shape[0]:
            raise ValueError(
                "Number of profits should be equal to size of thetas matrix"
            )
        else:
            self.thetas = thetas

    def generate_theta(self) -> np.ndarray:
        """Generates random theta matrix. All diagonal elements have mean 1 and standard deviation of 0,2. Every other element have mean 0 and std 0,2

        Returns:
            np.ndarray: thetas matrix
        """
        out = np.random.normal(0, 0.2, size=(self.n_products, self.n_products))
        for i in range(self.n_products):
            out[i, i] += 1
        return out

    def get_reward(self, assesment: list[int]):
        """get reward and demand based on given assesment/placement/x. To do this, will sample random demand vector from lognormal distribution.

        Args:
            assesment (list[int]): list's size is number of products. Values are 1 for chosen products and 0 for not chosen.

        Returns 2 elements: :  
            int: reward or profit of given input
            np.ndarray: demand vector that was used to compute profit
        """
        x = np.array(assesment)

        # vector of demand's means
        demand_means_vector = self.thetas @ x  # (n, n) x (n, 1) -> (n, 1)
        demand_out = np.ndarray(shape=self.n_products)
        total_profit = 0
        for i in range(len(assesment)):
            # sample demand for item `i` based on their mean and global demand std
            curent_demand = np.exp(
                norm.ppf(np.random.uniform(), demand_means_vector[i], self.demand_std)
            )
            demand_out[i] = curent_demand
            total_profit += self.profits[i] * assesment[i] * curent_demand
        return total_profit, demand_out
