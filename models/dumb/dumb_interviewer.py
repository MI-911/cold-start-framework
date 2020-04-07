from models.base_interviewer import InterviewerBase
from shared.meta import Meta


class DumbInterviewer(InterviewerBase):
    def __init__(self, meta: Meta, recommender=None, use_cuda=False):
        super().__init__(meta, use_cuda)

        if not recommender:
            raise RuntimeError('No underlying recommender provided to the dumb interviewer.')

        self.recommender = recommender(meta)

    def predict(self, items, answers):
        return self.recommender.predict(items, answers)

    def interview(self, answers, max_n_questions=5):
        return []

    def warmup(self, training, interview_length=5):
        self.recommender.fit(training)

    def get_parameters(self):
        pass

    def load_parameters(self, parameters):
        pass
