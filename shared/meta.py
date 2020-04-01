from typing import Dict, List

from shared.graph_triple import GraphTriple


class Meta:
    def __init__(self, entities: Dict[str, Dict], uri_idx: Dict[str, int], idx_item: Dict[int, bool],
                 users: List[int], recommendable_entities: List[int], triples: List[GraphTriple],
                 default_cutoff: int):
        self.entities = entities
        self.uri_idx = uri_idx
        self.idx_item = idx_item
        self.users = users
        self.recommendable_entities = recommendable_entities
        self.triples = triples
        self.default_cutoff = default_cutoff
