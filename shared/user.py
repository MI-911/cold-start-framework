class User:
    """
    Base class for users. All users, whether cold or warm, have a validation set.
    """
    def __init__(self, validation):
        self.validation = validation


class WarmStartUser(User):
    """
    For the warm-up method of recommenders.
    """
    def __init__(self, ratings, validation):
        """
        Initialises a warm-start user.
        :param ratings: A dictionary of the format: { entity_id: sentiment }
        :param validation: A pair of the format: (positive, negatives)
        """
        super().__init__(validation)
        self.training = ratings


class ColdStartUserSet:
    def __init__(self, answers, positive, negative):
        """
        Initialises a cold-start user set.
        :param answers: A dictionary of the format: { entity_id: sentiment }
        :param positive: An entity_id
        :param negative: A list of the format [ entity_id ]
        """
        self.answers = answers
        self.positive = positive
        self.negative = negative


class ColdStartUser(User):
    """
    For the interview and predict methods of recommenders.
    """
    def __init__(self, sets: ColdStartUserSet, validation):
        """
        Initialises a cold-start user.
        :param sets: A list of the format [ ColdStartUserSet ]
        :param validation: A pair of the format (positive, negatives)
        """
        super().__init__(validation)
        self.sets = sets
