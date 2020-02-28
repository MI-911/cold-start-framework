

class User:
    """
    The warmup() method in Recommenders take a list of User objects as input.
    """
    def __init__(self, ratings, validation):
        self.training = ratings  # { entity_id: sentiment }
        self.validation = validation  # [ (positives, negative) ]
