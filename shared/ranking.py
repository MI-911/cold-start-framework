from typing import List


class Ranking:
    def __init__(self, to_rank: List[int], positives: List[int]):
        self.to_rank = to_rank
        self.positives = positives

        # Assert that there are no duplicates
        assert len(set(self.to_rank)) == len(self.to_rank)

        # Assert that all positives in the ranked list
        assert len(set(self.to_rank)) == len(set(self.to_rank).union(set(self.positives)))
