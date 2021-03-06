from typing import Dict, List

from shared.meta import Meta
from shared.user import WarmStartUser


class InterviewerBase:
    def __init__(self, meta: Meta, use_cuda=False):
        self.meta = meta
        self.use_cuda = use_cuda

    def warmup(self, training: Dict[int, WarmStartUser], interview_length=5) -> None:
        """
        Fits the model to the training data.
        :param training: A dictionary of the following format:
            {
                userId : {
                    training: {
                        entityId: sentiment
                    },
                    validation: {
                        negative: [ item_ids ],
                        positive: item_id
                    }
                }
            }
        :param interview_length: The length of the interview that should be learned.
        :return: None
        """
        raise NotImplementedError

    def interview(self, answers: Dict, max_n_questions=5) -> List[int]:
        """
        Predicts the next question(s) to ask a user
        :param answers: A state dictionary of the following format:
            {
                item_id: sentiment
            }
        :param max_n_questions: The maximally allowed length of the returned list.
        :return: A list of the following format:
            [
                item_ids  // Next question(s) to ask the user
            ]
        """
        raise NotImplementedError

    def predict(self, items: List[int], answers: Dict) -> Dict[int, float]:
        """
        Predicts the ranking of the provided items given a user's answers.
        :param items: A list of the following format:
            [
                item_ids  // The items to rank
            ]
        :param answers: A state dictionary of the following format:
            {
                item_id: sentiment
            }
        :return: A dictionary of the following format:
            {
                item_id: score  // The best fitting item gets the highest score.
            }
        """
        raise NotImplementedError

    def get_parameters(self):
        """
        Returns the model's current parameters so they can be reused later.
        :return: A dictionary of { param_key: value } key-value pairs.
        """
        raise NotImplementedError

    def load_parameters(self, params):
        """
        Loads the provided model parameters. Parameters should be provided from an outside source (e.g. the runner).
        :param params: A dictionary of { param_key: value } key-value pairs.
        """
        raise NotImplementedError

