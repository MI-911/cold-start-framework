from enum import Enum


class EntityType(Enum):
    RECOMMENDABLE = 1
    DESCRIPTIVE = 2
    ANY = 3


class Sentiment(Enum):
    NEGATIVE = 1
    UNKNOWN = 2
    POSITIVE = 3
    UNSEEN = 4
    ANY = 5
