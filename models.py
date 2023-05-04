from abc import ABC, abstractmethod
from itertools import combinations

import numpy as np
from scipy.stats import beta, norm

from rewards import AbstractReward, BernouliBanditReward, ProductAssortmentReward


class Solver(ABC):
    """Abstract class for solving multi-arm bandit problems"""

    def __init__(self, k: int, reward: AbstractReward):
        """Abstract class for solving multi-arm bandit problems

        Args:
            k (int): number of arms

            reward (AbstractReward): reward of problem
        """
        self.k = k
        self.reward = reward

    @abstractmethod
    def iterate(self) -> list[int, float]:
        """abstract method for one iteration of algorithm

        Returns:
            list[int, float]: action choosen and reward
        """
        pass

    def simulate(self, n_iter: int, normalise=True):
        """Simulate algorithm for given number of iteration, will return action proportions and rewards.

        Action proportions is np.ndarray with shape (k, n_iter). Where first index shows how many times action was used from 1 iteration to second index.
        If normalise=True, then every action element will be normalised based on total number of actions in iteration.

        rewards is np.ndarray with shape (n_iter). It just contains reward on every iteration.

        Args:
            n_iter (int): number of iterations to simulate

            normalise (bool, optional): Should normalise. Defaults to True.

        Returns:
            (np.ndarray, np.ndarray): tuple of two elements: actions array and rewards
        """
        actions = np.zeros(shape=(self.k, n_iter + 1))
        total_reward = np.zeros(shape=n_iter)
        for i in range(n_iter):
            cur_action, c_reward = self.iterate()
            for j in range(len(actions)):
                actions[j][i + 1] = actions[j][i]
                total_reward[i] = c_reward
            actions[cur_action][i + 1] += 1
        if normalise:
            for i in range(1, n_iter + 1):
                actions[:, i] /= np.sum(actions[:, i])
        return actions[:, 1:], total_reward


class BetaGreedyBandit(Solver):
    """Greedy approach to Bernoulli multi-arm bandit with Beta distribution as prior."""

    def __init__(self, k: int, reward: BernouliBanditReward):
        """

        Args:
            k (int): number of arms

            reward (BernouliBanditReward): reward
        """
        super().__init__(k, reward)
        # Prior assumption is uniform distribution
        self.args = [[1, 1] for _ in range(k)]

    def iterate(self):
        action = self.get_action()
        result = self.reward.get_reward(action)
        # update priors based on conjugate properties of Beta distribution
        self.args[action][0] += int(result)
        self.args[action][1] += 1 - int(result)
        return action, result

    def get_thetas(self) -> list[float]:
        """Returns expected value of theta for each arm.

        Returns:
            list[float]: array of thetas for each arm
        """
        thetas = []
        for a, b in self.args:
            thetas.append(a / (a + b))
        return thetas

    def get_action(self):
        thetas = self.get_thetas()
        return np.argmax(thetas)


class ThompsonBernouliBandit(Solver):
    """Thompson sampling for multi-arm bandit problem with Bernoulli reward using Beta distribution as prior"""

    def __init__(self, k: int, reward: BernouliBanditReward):
        # Prior assumption if uniform
        self.args = [[1, 1] for _ in range(k)]
        super().__init__(k, reward)

    def iterate(self):
        action = np.argmax(
            self.get_thetas(),
        )
        result = self.reward.get_reward(action)
        # update priors based on conjugate properties of Beta distribution
        self.args[action][0] += int(result)
        self.args[action][1] += 1 - int(result)
        return action, result

    def get_thetas(self) -> list[float]:
        """return list of thetas for each arm, which are sampled from their prior distribution.

        Returns:
            list[float]: list of thetas for each arm
        """
        thetas = []
        for a, b in self.args:
            thetas.append(beta.ppf(np.random.uniform(), a, b))
        return thetas


class UCBsolver(Solver):
    """Upper Confidence Bound (UCB) approach for multi-arm bandit problem. See more https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning"""

    def __init__(self, k: int, reward: AbstractReward, confidence_level: float):
        """

        Args:
            k (int): number of arms

            reward (AbstractReward)

            confidence_level (float): coefficient that controls exploration and explotation. The bigger, the more explorative
        """
        super().__init__(k, reward)
        self.n_calls = [0 for _ in range(k)]
        self.means = [0 for _ in range(k)]
        self.confidence_level = confidence_level
        self.count = 0

    def get_thetas(self) -> list[float]:
        """Get thetas based on their mean plus uncertainty part. See for more https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning.

        Returns:
            list[float]: thetas for every arm
        """
        values = []
        for i, mean in enumerate(self.means):
            # if called first time give infinity, because of division by zero
            if self.n_calls[i] == 0:
                values.append(np.inf)
            else:
                uncertainty = self.confidence_level * np.sqrt(
                    np.log(1.5 * self.count) / self.n_calls[i]
                )
                values.append(mean + uncertainty)
        return values

    def iterate(self) -> int:
        action = self.get_action()
        result = self.reward.get_reward(action)
        # update mean
        self.means[action] = self.means[action] * self.n_calls[action] + result
        self.n_calls[action] += 1
        self.means[action] /= self.n_calls[action]
        self.count += 1
        return action, result

    def get_action(self) -> int:
        i = np.argmax(self.get_thetas())
        return i


class GaussianThompson(Solver):
    """Thompson sampling for multi-arm bandit problem with any reward with Gaussian prior and known variances."""

    def __init__(self, k: int, variances: list[float], reward: AbstractReward):
        """

        Args:
            k (int): number of arms

            variances (list[float]): variances of rewards for every arm

            reward (AbstractReward)
        """
        super().__init__(k, reward)
        self.variances = variances

        # priors are assumed normal with mean 0 and std 10000.
        self.priors = [[0, 10000] for _ in range(k)]
        self.rewards_sum = [0 for _ in range(k)]
        self.n_calls = [0 for _ in range(k)]

    def iterate(self) -> int:
        action = self.get_action()
        result = self.reward.get_reward(action)
        self.rewards_sum[action] += result
        self.n_calls[action] += 1
        self.update(action)
        return action, result

    def update(self, action: int) -> None:
        """update prior distribution based on https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution

        Args:
            action (int): which action was chosen
        """
        old_mu = self.priors[action][0]
        old_var = self.priors[action][1]
        n = self.n_calls[action]
        self.priors[action][1] = 1 / (1 / old_var + n / self.variances[action])
        self.priors[action][0] = (
            old_mu / old_var + self.rewards_sum[action] / self.variances[action]
        ) / (1 / old_var + n / self.variances[action])

    def get_action(self) -> int:
        values = []
        for mu, variance in self.priors:
            values.append(norm.ppf(np.random.uniform(), mu, np.sqrt(variance)))
        return np.argmax(values)


class AssortmentSolver:
    """Abstract class for Product Assortment problem with log-normal prior. See more https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf, page 51"""

    def __init__(
        self,
        n: int,
        k: int,
        profits: list[float],
        reward: ProductAssortmentReward,
        demand_std: float,
        prior_means=None,
        prior_covariance_matrix=None,
    ) -> None:
        """

        Args:
            n (int): number of products

            k (int): number of products that should be chosen

            profits (list[float]): number of products that should be picked

            reward (ProductAssortmentReward)

            demand_std (float): standart deviation of demand

            prior_means (_type_, optional): prior mean assumption, shape (n*n). Defaults to None.

            prior_covariance_matrix (_type_, optional): prior covariance matrix, shape (n*n, n*n). Defaults to None.
        """
        self.k = k
        self.n = n
        self.profits = profits
        self.reward = reward
        self.demand_std = demand_std
        if prior_means is None:
            self.priors_means = np.zeros(shape=n * n)
        else:
            self.priors_means = prior_means

        if prior_covariance_matrix is None:
            self.covariance_matrix = np.full((n * n, n * n), 0.2)
            for i in range(n * n):
                self.covariance_matrix[i, i] = 1
        else:
            self.covariance_matrix = prior_covariance_matrix

    def get_expected(self, x: np.ndarray, theta: np.ndarray) -> float:
        """compute expected profit for given x and prior theta

        Args:
            x (np.ndarray): shape (n), 0 or 1 values, number of 1 is equal k

            theta (np.ndarray): shape (n, n)

        Returns:
            float: expected profit
        """
        total_profit = 0
        total_demand = np.exp(theta @ x + (self.demand_std**2) / 2)
        for i in range(self.n):
            expected_demand = total_demand[i]
            total_profit += self.profits[i] * x[i] * expected_demand
        return total_profit

    def get_thetas(self) -> np.ndarray:
        """abstract method for getting theta

        Returns:
            np.ndarray: theta with shape (n, n)
        """
        raise NotImplementedError()

    def iterate(self, count_optimal=False) -> list[int]:
        """make one iteration of algorithm.

        Args:
            count_optimal (bool, optional): Should we count optimal solution. Defaults to False.

        Returns:
            list[int]: if count_optimal=False, then list of (best x, it's reward), else list of (best x, it's reward, optimal reward)
        """
        if count_optimal:
            optimal_reward = -np.inf

        # save best x based on expected profit
        best_x = None
        best_expected = None
        best_expected_reward = None
        best_expected_demand = None
        sample_theta = self.get_thetas()
        # Compute all possible combinations of k elements from n, order does not matter
        for inds in combinations([i for i in range(self.n)], self.k):
            x = np.zeros(shape=self.n)
            x[list(inds)] = 1

            if count_optimal:
                real_reward, real_demand = self.reward.get_reward(x)
                if real_reward > optimal_reward:
                    optimal_reward = real_reward

            # get expected profit
            cur_expected = self.get_expected(x, sample_theta)
            if best_expected is None or cur_expected > best_expected:
                best_expected = cur_expected
                best_x = np.copy(x)
                if count_optimal:
                    best_expected_reward = real_reward
                    best_expected_demand = real_demand

        if count_optimal:
            reward, real_demand = best_expected_reward, best_expected_demand
        else:
            reward, real_demand = self.reward.get_reward(best_x)

        self.update(best_x, real_demand)

        if count_optimal:
            return best_x, reward, optimal_reward
        else:
            return best_x, reward

    def update(self, x: np.ndarray, demand: np.ndarray):
        """update prior distributions. To see math behind: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf, page 51

        Args:
            x (np.ndarray): chosen placement

            demand (np.ndarray): real demand on that placement
        """
        z = np.log(demand[x == 1])  # k, 1
        s = np.zeros(shape=(self.k, self.n))  # k, n
        cnt = 0
        for i in range(self.n):
            if x[i] != 0:
                s[cnt, i] = 1
                cnt += 1
        x.shape = (x.shape[0], 1)  # n, 1
        w = np.kron(x.T, s)  # k, n^2
        old_cov = np.copy(self.covariance_matrix)
        old_means = np.copy(self.priors_means)
        cov_inverse = np.linalg.inv(old_cov)
        self.covariance_matrix = np.linalg.inv(
            cov_inverse + w.T @ w / self.demand_std**2
        )
        self.priors_means = self.covariance_matrix @ (
            cov_inverse @ old_means + w.T @ z / self.demand_std**2
        )

    def simulate(self, n_iter=1000) -> list[np.ndarray]:
        """Simulate algorithm for given number of iterations. Similar to Solver(abc)

        Args:
            n_iter (int, optional): number of iterations. Defaults to 1000.

        Returns:
            list[np.ndarray]: list of 2 elements: [rewards per iteration, optimal reward per iteration]
        """
        rewards = np.ndarray(shape=n_iter)
        optimals = np.ndarray(shape=n_iter)
        for i in range(n_iter):
            _, reward, optimal = self.iterate(count_optimal=True)
            rewards[i] = reward
            optimals[i] = optimal
        return rewards, optimals


class ThompsonAssortmnet(AssortmentSolver):
    """Thompson implementation of Product Assortment problem"""

    def get_thetas(self) -> np.ndarray:
        """Sample theta based on their prior distribution

        Returns:
            np.ndarray: (n, n) array of thetas
        """
        thetas = np.random.multivariate_normal(
            self.priors_means, self.covariance_matrix
        )
        return np.reshape(thetas, (self.n, self.n))


class GreedyAssortment(AssortmentSolver):
    """Greedy approach for Product Assortment Problem"""

    def get_thetas(self) -> np.ndarray:
        """Get theta as expected value of their prior distribution

        Returns:
            np.ndarray: (n, n) array of thetas
        """
        out = np.reshape(self.priors_means, (self.n, self.n))
        return out
