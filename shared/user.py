from typing import Dict, List

from shared.ranking import Ranking


class WarmStartUser:
    """
    For the warm-up method of recommenders.
    """
    def __init__(self, ratings: Dict[int, int], validation: Ranking):
        """
        Initialises a warm-start user.
        :param ratings: A dictionary of the format: { entity_id: sentiment }
        :param validation: The ranking used for validation.
        """
        self.training = ratings
        self.validation = validation


class ColdStartUserSet:
    def __init__(self, answers, ranking: Ranking):
        """
        Initialises a cold-start user set.
        :param answers: A dictionary of the format: { entity_id: sentiment }
        :param ranking: The ranking used for evaluation.
        """
        self.answers = answers
        self.ranking = ranking


class ColdStartUser:
    """
    For the interview and predict methods of recommenders.
    """
    def __init__(self, sets: List[ColdStartUserSet]):
        """
        Initialises a cold-start user.
        :param sets: A list of the format [ ColdStartUserSet ]
        """
        self.sets = sets

