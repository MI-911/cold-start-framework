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

        # Live parameters set when running the interviewers, should not be changed here
        self.recommendable_only = False
        self.type_limit = None
        self.popular_items = None

    def get_idx_uri(self):
        return {idx: uri for uri, idx in self.uri_idx.items()}

    def get_question_candidates(self, training: Dict[int, WarmStartUser], limit: int = None,
                                recommendable_only: bool = None):
        """
        Get entity index candidates for interviews. The flag recommendable_only on the meta instance determines whether
        only recommendable entities are allowed, but can be overwritten through the function parameter.
        """
        candidates = get_top_entities(training)
        idx_uri = self.get_idx_uri()
        idx_labels = {idx: set([label.lower() for label in self.entities[idx_uri[idx]]['labels']]) for idx in idx_uri}

        recommendable_only = self.recommendable_only if recommendable_only is None else recommendable_only
        if recommendable_only:
            candidates = [entity for entity in candidates if entity in self.recommendable_entities]
        else:
            if self.type_limit:
                type_limit = set(self.type_limit)

                candidates = [entity for entity in candidates if type_limit.intersection(idx_labels[entity])]
            else:
                candidates = [entity for entity in candidates if entity not in self.recommendable_entities]

        if limit:
            candidates = candidates[:limit]

        return candidates
