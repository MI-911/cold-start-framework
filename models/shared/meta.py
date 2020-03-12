from typing import Dict, List


class Meta:
    def __init__(self, entities: Dict[str, Dict], uri_idx: Dict[str, int], idx_item: Dict[int, bool],
                 users: List[int], recommendable_entities: List[int]):
        self.entities = entities
        self.uri_idx = uri_idx
        self.idx_item = idx_item
        self.users = users
        self.recommendable_entities = recommendable_entities
