import numpy as np
from loguru import logger
from scipy import stats
from typing import List, Tuple


def average_precision(ranked_relevancy_list):
    """
    Calculates the average precision AP@k. In this setting, k is the length of
    ranked_relevancy_list.
    :param ranked_relevancy_list: A one-hot numpy list mapping the recommendations to
    1 if the recommendation was relevant, otherwise 0.
    :return: AP@k
    """

    if len(ranked_relevancy_list) == 0:
        a_p = 0.0
    else:
        p_at_k = ranked_relevancy_list * np.cumsum(ranked_relevancy_list, dtype=np.float32) / (1 + np.arange(ranked_relevancy_list.shape[0]))
        a_p = np.sum(p_at_k) / ranked_relevancy_list.shape[0]

    assert 0 <= a_p <= 1, a_p
    return a_p


def dcg(rank, n=10):
    r = np.zeros(n)
    if rank < n:
        r[rank] = 1

    return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))


def tau_at_k(utility, k):
    # Cutoff must match length of utility list
    if len(utility) != k:
        return np.NaN

    tau, p = stats.kendalltau(utility[:k], sorted(utility, reverse=True)[:k])

    return tau


def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ser_at_k(ranked_labeled_items, top_pop_items, k, normalize=False):
    serendipitous_labeled_items = [
        (item, relevance)
        for item, relevance in ranked_labeled_items
        if item not in top_pop_items[:k]
    ]

    if l := len(serendipitous_labeled_items) == 0:
        return 0

    return sum([relevance for item, relevance in serendipitous_labeled_items]) / (
        l if normalize else 1
    )


def ser_at_k_v2(ranked_labeled_items: List[Tuple[int, int]], n_ratings: np.ndarray, k: int, normalize=False): 
    ranked_labeled_items = list(ranked_labeled_items)
    items = [i for i, _ in ranked_labeled_items]
    top_pop = list(sorted([(i, rs) for i, rs in zip(items, n_ratings[items])], key=lambda x: x[1], reverse=True))
    top_pop = [i for i, rs in top_pop]

    serendipitous_labeled_items = [
        (item, relevance)
        for item, relevance in ranked_labeled_items
        if item not in top_pop[:k]
    ]

    if l := len(serendipitous_labeled_items) == 0:
        return 0

    return sum([relevance for item, relevance in serendipitous_labeled_items]) / (
        l if normalize else 1
    )


def coverage(recommended_entities, recommendable_entities):
    return len(recommended_entities) / len(recommendable_entities)


def hr_at_k(relevance, cutoff):
    return 1 in relevance[:cutoff]


if __name__ == '__main__':
    logger.info(ndcg_at_k([0, 0, 1.0], 3))
    logger.info(ndcg_at_k([0, 0, 1], 3))
