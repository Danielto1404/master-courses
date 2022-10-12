import numpy as np
import scipy.stats as stats


class Improvement:

    def __init__(self):
        self.iteration = 1

    def score(self, mu: float, sigma: float, **kwargs) -> float:
        raise NotImplementedError("`score` method must be implemented in subclass")

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def step(self):
        self.iteration += 1


class ConfidenceBound(Improvement):
    __MODES__ = ["upper", "lower"]
    __DECAYS__ = ["const", "linear", "log2", "sqrt", "squared"]

    def __init__(self, mode="upper", alpha=1.0, decay="const"):
        assert mode in ConfidenceBound.__MODES__, \
            f"Unknown mode variant, please use of of this: {ConfidenceBound.__MODES__}"

        assert decay in ConfidenceBound.__DECAYS__, \
            f"Unknown decay variant, please use of of this: {ConfidenceBound.__DECAYS__}"

        self.mode = mode
        self.alpha = alpha
        self.decay = decay

        super().__init__()

    def score(self, mu: float, sigma: float, **kwargs) -> float:
        sign = 1 if self.mode == "upper" else -1
        return -(mu + sign * self.alpha * self.coef * sigma)

    @property
    def coef(self) -> float:
        if self.decay == "linear":
            scale = self.iteration
        elif self.decay == "log2":
            scale = np.log2(self.iteration + 1)
        elif self.decay == "sqrt":
            scale = np.sqrt(self.iteration)
        elif self.decay == "squared":
            scale = self.iteration ** 2
        else:
            scale = 1

        return 1.0 / scale

    def __repr__(self):
        return f"{self.__class__.__name__}(mode={self.mode}, alpha={self.alpha}, decay={self.decay})"


class ProbabilityImprovement(Improvement):
    def __init__(self, slack: float = 0):
        super().__init__()
        self.slack = slack

    def score(self, mu: float, sigma: float, **kwargs) -> float:
        best_score = kwargs.get("best_score", -np.inf)
        z = (best_score - mu - self.slack) / sigma
        return stats.norm().cdf(z)

    def __repr__(self):
        return f"{self.__class__.__name__}(slack={self.slack})"


class ExpectedImprovement(Improvement):
    """
    Notes: http://ash-aldujaili.github.io/blog/2018/02/01/ei/
    """

    def __init__(self, slack: float = 0):
        super().__init__()
        self.slack = slack

    def score(self, mu: float, sigma: float, **kwargs) -> float:
        best_score = kwargs.get("best_score", -np.inf)

        d = best_score - mu - self.slack
        z = d / sigma

        cdf = stats.norm().cdf(z)
        pdf = stats.norm().pdf(z)

        return d * z * cdf + sigma * pdf

    def __repr__(self):
        return f"{self.__class__.__name__}(slack={self.slack})"


class ThompsonImprovement(Improvement):
    def score(self, mu: float, sigma: float, **kwargs) -> float:
        return sigma


__all__ = [
    "Improvement",
    "ConfidenceBound",
    "ProbabilityImprovement",
    "ExpectedImprovement",
    "ThompsonImprovement"
]
