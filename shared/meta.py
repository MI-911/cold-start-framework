from typing import Dict, List

from shared.enums import Sentiment
from shared.graph_triple import GraphTriple
from shared.user import WarmStartUser
from shared.utility import get_top_entities
from shared.validator import Validator


class Meta:
    def __init__(self, entities: Dict[str, Dict], uri_idx: Dict[str, int], idx_item: Dict[int, bool],
                 users: List[int], recommendable_entities: List[int], triples: List[GraphTriple],
                 default_cutoff: int, sentiment_utility: Dict[Sentiment, float],
                 validator: Validator = None):
        self.entities = entities
        self.uri_idx = uri_idx
        self.idx_item = idx_item
        self.users = users
        self.recommendable_entities = recommendable_entities
        self.triples = triples
        self.default_cutoff = default_cutoff
        self.sentiment_utility = sentiment_utility
        self.validator = validator
        self.recommendable_only = False

    def get_idx_uri(self):
        return {idx: uri for uri, idx in self.uri_idx.items()}

    def get_question_candidates(self, training: Dict[int, WarmStartUser], limit: int = None):
        """
        Get entity index candidates for interviews. The flag recommendable_only on the meta instance determines whether
        only recommendable entities are allowed.
        """
        candidates = get_top_entities(training)

        if self.recommendable_only:
            candidates = [entity for entity in candidates if entity in self.recommendable_entities]

        if limit:
            candidates = candidates[:limit]

        return candidates
