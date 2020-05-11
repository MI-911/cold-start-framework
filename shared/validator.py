import operator
from statistics import mean
from typing import List, Tuple, Dict

from experiments.metrics import coverage, hr_at_k, ndcg_at_k, tau_at_k
from shared.enums import Metric
from shared.ranking import Ranking


class Validator:
    def __init__(self, metric: Metric, cutoff: int = 10):
        self.metric = metric
        self.cutoff = cutoff

    def score(self, predictions: List[Tuple[Ranking, Dict[int, float]]], meta, reverse=True,
              metric: Metric = None, cutoff: int = None) -> float:
        covered = set()
        scores = list()

        # By default, use the metric and cutoff defined in the validator
        metric = metric if metric else self.metric
        cutoff = cutoff if cutoff else self.cutoff

        for ranking, item_scores in predictions:
            # Convert scores of item->score pairs to a ranked list
            sorted_scores = sorted(item_scores.items(), key=operator.itemgetter(1), reverse=reverse)
            prediction = [pair[0] for pair in sorted_scores]

            if metric == Metric.COV:
                covered.update(set(prediction[:cutoff]))
            elif metric == Metric.HR:
                scores.append(hr_at_k(ranking.get_relevance(prediction), cutoff))
            elif metric == Metric.NDCG:
                scores.append(ndcg_at_k(ranking.get_utility(prediction, meta.sentiment_utility), cutoff))
            elif metric == Metric.TAU:
                scores.append(tau_at_k(ranking.get_utility(prediction, meta.sentiment_utility), cutoff))
            else:
                raise RuntimeError('Unsupported metric for validation.')

        if metric == Metric.COV:
            return coverage(covered, meta.recommendable_entities)
        else:
            return mean(scores)
