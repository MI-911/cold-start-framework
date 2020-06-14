import operator
from statistics import mean
from typing import List, Tuple, Dict

from experiments.metrics import coverage, hr_at_k, ndcg_at_k, tau_at_k, ser_at_k
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
            prediction = sorted(item_scores, key=item_scores.get, reverse=reverse)
            ranked_cutoff = prediction[:cutoff]
            relevance_cutoff = ranking.get_relevance(ranked_cutoff)

            if metric == Metric.COV:
                covered.update(set(prediction[:cutoff]))
            elif metric == Metric.HR:
                scores.append(hr_at_k(ranking.get_relevance(prediction), cutoff))
            elif metric == Metric.NDCG:
                scores.append(ndcg_at_k(ranking.get_utility(prediction, meta.sentiment_utility), cutoff))
            elif metric == Metric.TAU:
                scores.append(tau_at_k(ranking.get_utility(prediction, meta.sentiment_utility), cutoff))
            elif metric == Metric.SER:
                scores.append(ser_at_k(zip(ranked_cutoff, relevance_cutoff), meta.popular_items, cutoff,
                                       normalize=False))
            elif metric == Metric.MIXED:
                score = list()

                score.append(ser_at_k(zip(ranked_cutoff, relevance_cutoff), meta.popular_items, cutoff, normalize=False))
                score.append(ndcg_at_k(ranking.get_utility(prediction, meta.sentiment_utility), cutoff))
                score.append(hr_at_k(ranking.get_relevance(prediction), cutoff))

                scores.append(mean(score))
            else:
                raise RuntimeError('Unsupported metric for validation.')

        if metric == Metric.MIXED:
            return mean([coverage(covered, meta.recommendable_entities), mean(scores)])
        elif metric == Metric.COV:
            return coverage(covered, meta.recommendable_entities)
        else:
            return mean(scores) if scores else 0
