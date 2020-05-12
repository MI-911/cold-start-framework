from typing import List, Dict

from models.base_interviewer import InterviewerBase
from shared.meta import Meta
from shared.user import WarmStartUser


class MonteCarloNetInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, use_cuda=False):
        super(MonteCarloNetInterviewer, self).__init__(meta)
        self.model = MonteCarloNet()

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:


    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        pass

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        pass

    def get_parameters(self):
        pass

    def load_parameters(self, params):
        pass