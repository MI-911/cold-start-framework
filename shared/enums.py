from enum import Enum, IntEnum


class UnseenSampling(Enum):
    UNIFORM = 1
    POPULARITY = 2
    EQUAL_POPULARITY = 3


class SeenSampling(Enum):
    STANDARD = 1
    LONG_TAIL = 2


class EntityType(Enum):
    RECOMMENDABLE = 1
    DESCRIPTIVE = 2
    ANY = 3


class Sentiment(IntEnum):
    NEGATIVE = 1
    UNKNOWN = 2
    POSITIVE = 3
    UNSEEN = 4
    ANY = 5


class Metric(Enum):
    HR = 1
    NDCG = 2
    SER = 3
    COV = 4
    TAU = 5
