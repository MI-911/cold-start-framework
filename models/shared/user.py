

class WarmStartUser:
    """
    For the warmup() method of recommenders.
    """
    def __init__(self, ratings, validation):
        """
        Initialises a warm-start user.
        :param ratings: A dictionary of the format: { entity_id: sentiment }
        :param validation: A pair of the format: (positive, negatives)
        """
        self.training = ratings
        self.validation = validation


class ColdStartUserSet:
    def __init__(self, answers, positive, negatives):
        """
        Initialises a cold-start user set.
        :param answers: A dictionary of the format: { entity_id: sentiment }
        :param positive: An entity_id
        :param negatives: A list of the format [ entity_id ]
        """
        self.answers = answers
        self.positive = positive
        self.negatives = negatives


class ColdStartUser:
    """
    For the interview() and predict() methods of Recommenders.
    """
    def __init__(self, sets, validation):
        """
        Initialises a cold-start user.
        :param sets: A list of the format [ ColdStartUserSet ]
        :param validation: A pair of the format (positive, negatives)
        """
        self.sets = sets
        self.validation = validation
