from models.ddpg.ddpg_interviewer import DDPGInterviewer
from models.dqn.dqn_new import DqnInterviewer
from models.dumb.dumb_interviewer import DumbInterviewer
from models.fmf.fmf_interviewer import FMFInterviewer
from models.greedy.greedy_interviewer import GreedyInterviewer
from models.lrmf.lrmf_interviewer import LRMFInterviewer
from models.melu.melu_interviewer import MeLUInterviewer
from models.naive.naive_interviewer import NaiveInterviewer
from recommenders.knn.knn_recommender import KNNRecommender
from recommenders.mf.mf_recommender import MatrixFactorizationRecommender
from recommenders.pagerank.collaborative_pagerank_recommender import CollaborativePageRankRecommender
from recommenders.pagerank.joint_linear_learned_pagerank_recommender import PairLinearJointPageRankRecommender
from recommenders.pagerank.joint_pagerank_recommender import JointPageRankRecommender
from recommenders.pagerank.kg_pagerank_recommender import KnowledgeGraphPageRankRecommender
from recommenders.pagerank.linear_combined_pagerank_recommender import LinearCombinedPageRankRecommender
from recommenders.pagerank.pair_linear_combined_pagerank_recommender import PairLinearCombinedPageRankRecommender
from recommenders.random.random_recommender import RandomRecommender
from recommenders.toppop.toppop_recommender import TopPopRecommender

models = {
    'ddpg-ppr-joint': {
        'interviewer': DDPGInterviewer,
        'recommender': JointPageRankRecommender,
        'requires_interview_length': True
    },
    'ddpg-ppr-collab': {
        'interviewer': DDPGInterviewer,
        'recommender': CollaborativePageRankRecommender,
        'requires_interview_length': True
    },
    'ddpg-ppr-kg': {
        'interviewer': DDPGInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender,
        'requires_interview_length': True
    },
    'ddpg-mf': {
        'interviewer': DDPGInterviewer,
        'recommender': MatrixFactorizationRecommender,
        'requires_interview_length': True
    },
    'ddpg-knn': {
        'interviewer': DDPGInterviewer,
        'recommender': KNNRecommender,
        'requires_interview_length': True
    },
    'random': {
        'interviewer': DumbInterviewer,
        'recommender': RandomRecommender
    },
    'top-pop': {
        'interviewer': DumbInterviewer,
        'recommender': TopPopRecommender
    },
    'greedy-ppr-kg': {
        'interviewer': GreedyInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender
    },
    'greedy-adaptive-ppr-linear-learned': {
        'interviewer': GreedyInterviewer,
        'recommender': PairLinearCombinedPageRankRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'greedy-adaptive-ppr-linear-joint-learned': {
        'interviewer': GreedyInterviewer,
        'recommender': PairLinearJointPageRankRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'greedy-adaptive-ppr-kg': {
        'interviewer': GreedyInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'greedy-adaptive-ppr-collab': {
        'interviewer': GreedyInterviewer,
        'recommender': CollaborativePageRankRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'greedy-ppr-collab': {
        'interviewer': GreedyInterviewer,
        'recommender': CollaborativePageRankRecommender
    },
    'greedy-ppr-joint': {
        'interviewer': GreedyInterviewer,
        'recommender': JointPageRankRecommender
    },
    'greedy-adaptive-ppr-joint-threeway': {
        'interviewer': GreedyInterviewer,
        'recommender': JointPageRankRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'greedy-adaptive-knn': {
        'interviewer': GreedyInterviewer,
        'recommender': KNNRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'greedy-adaptive-mf': {
        'interviewer': GreedyInterviewer,
        'recommender': MatrixFactorizationRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'lrmf': {
        'interviewer': LRMFInterviewer,
        'requires_interview_length': True,
        'use_cuda': False
    },
    'pop-ppr-collab': {
        'interviewer': NaiveInterviewer,
        'recommender': CollaborativePageRankRecommender
    },
    'pop-ppr-kg': {
        'interviewer': NaiveInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender
    },
    'pop-ppr-joint': {
        'interviewer': NaiveInterviewer,
        'recommender': JointPageRankRecommender
    },
    'pop-ppr-linear-learned': {
        'interviewer': NaiveInterviewer,
        'recommender': PairLinearCombinedPageRankRecommender
    },
    'pop-ppr-linear-joint-learned': {
        'interviewer': NaiveInterviewer,
        'recommender': PairLinearJointPageRankRecommender
    },
    'greedy-ppr-linear-joint-learned': {
        'interviewer': GreedyInterviewer,
        'recommender': PairLinearJointPageRankRecommender
    },
    'greedy-ppr-linear-learned': {
        'interviewer': GreedyInterviewer,
        'recommender': PairLinearCombinedPageRankRecommender
    },
    'greedy-knn': {
        'interviewer': GreedyInterviewer,
        'recommender': KNNRecommender
    },
    'pop-knn': {
      'interviewer': NaiveInterviewer,
      'recommender': KNNRecommender
    },
    'dqn-knn': {
        'interviewer': DqnInterviewer,
        'recommender': KNNRecommender,
        'requires_interview_length': True
    },
    'pop-mf': {
        'interviewer': NaiveInterviewer,
        'recommender': MatrixFactorizationRecommender
    },
    'greedy-mf': {
        'interviewer': GreedyInterviewer,
        'recommender': MatrixFactorizationRecommender
    },
    'dqn-mf': {
        'interviewer': DqnInterviewer,
        'recommender': MatrixFactorizationRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-kg': {
        'interviewer': DqnInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-collab': {
        'interviewer': DqnInterviewer,
        'recommender': CollaborativePageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-joint': {
        'interviewer': DqnInterviewer,
        'recommender': JointPageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'fmf': {
        'interviewer': FMFInterviewer,
        'requires_interview_length': True,
    },
    'melu': {
        'interviewer': MeLUInterviewer,
        'interviewer_kwargs': {
            'use_cuda': False,
            'use_sparse': False
        }
    }
}