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
from recommenders.pagerank.joint_pagerank_recommender import JointPageRankRecommender
from recommenders.pagerank.kg_pagerank_recommender import KnowledgeGraphPageRankRecommender
from recommenders.pagerank.linear_combined_pagerank_recommender import LinearCombinedPageRankRecommender
from recommenders.pagerank.linear_joint_pagerank_recommender import LinearJointPageRankRecommender
from recommenders.pagerank.linear_kg_pagerank_recommender import LinearKGPageRankRecommender
from recommenders.pagerank.pair_linear_combined_pagerank_recommender import PairLinearCombinedPageRankRecommender
from recommenders.random.random_recommender import RandomRecommender
from recommenders.toppop.toppop_recommender import TopPopRecommender

models = {
    'ddpg-ppr-joint': {
        'class': DDPGInterviewer,
        'recommender': JointPageRankRecommender,
        'requires_interview_length': True
    },
    'ddpg-ppr-collab': {
        'class': DDPGInterviewer,
        'recommender': CollaborativePageRankRecommender,
        'requires_interview_length': True
    },
    'ddpg-ppr-kg': {
        'class': DDPGInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender,
        'requires_interview_length': True
    },
    'ddpg-mf': {
        'class': DDPGInterviewer,
        'recommender': MatrixFactorizationRecommender,
        'requires_interview_length': True
    },
    'ddpg-knn': {
        'class': DDPGInterviewer,
        'recommender': KNNRecommender,
        'requires_interview_length': True
    },
    'random': {
        'class': DumbInterviewer,
        'recommender': RandomRecommender
    },
    'top-pop': {
        'class': DumbInterviewer,
        'recommender': TopPopRecommender
    },
    'greedy-ppr-kg': {
        'class': GreedyInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender
    },
    'greedy-adaptive-ppr-pair-linear': {
        'class': GreedyInterviewer,
        'recommender': PairLinearCombinedPageRankRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        },
        'recommender_kwargs': {
            'ask_limit': 10
        }
    },
    'greedy-adaptive-ppr-kg': {
        'class': GreedyInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'greedy-adaptive-ppr-collab': {
        'class': GreedyInterviewer,
        'recommender': CollaborativePageRankRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'greedy-ppr-collab': {
        'class': GreedyInterviewer,
        'recommender': CollaborativePageRankRecommender
    },
    'greedy-ppr-joint': {
        'class': GreedyInterviewer,
        'recommender': JointPageRankRecommender
    },
    'greedy-adaptive-ppr-joint': {
        'class': GreedyInterviewer,
        'recommender': JointPageRankRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'greedy-adaptive-knn': {
        'class': GreedyInterviewer,
        'recommender': KNNRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'greedy-adaptive-mf': {
        'class': GreedyInterviewer,
        'recommender': MatrixFactorizationRecommender,
        'interviewer_kwargs': {
            'adaptive': True
        }
    },
    'lrmf': {
        'class': LRMFInterviewer,
        'requires_interview_length': True,
        'use_cuda': False
    },
    'pop-ppr-collab': {
        'class': NaiveInterviewer,
        'recommender': CollaborativePageRankRecommender
    },
    'pop-ppr-kg': {
        'class': NaiveInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender
    },
    'pop-ppr-joint': {
        'class': NaiveInterviewer,
        'recommender': JointPageRankRecommender
    },
    'pop-ppr-linear-grid': {
        'class': NaiveInterviewer,
        'recommender': LinearCombinedPageRankRecommender
    },
    'pop-ppr-linear-learned': {
        'class': NaiveInterviewer,
        'recommender': PairLinearCombinedPageRankRecommender,
        'recommender_kwargs': {
            'ask_limit': 10
        }
    },
    'greedy-ppr-linear-grid': {
        'class': NaiveInterviewer,
        'recommender': LinearCombinedPageRankRecommender
    },
    'pop-knn': {
      'class': NaiveInterviewer,
      'recommender': KNNRecommender
    },
    'greedy-knn': {
        'class': GreedyInterviewer,
        'recommender': KNNRecommender
    },
    'dqn-knn': {
        'class': DqnInterviewer,
        'recommender': KNNRecommender,
        'requires_interview_length': True
    },
    'pop-mf': {
        'class': NaiveInterviewer,
        'recommender': MatrixFactorizationRecommender
    },
    'greedy-mf': {
        'class': GreedyInterviewer,
        'recommender': MatrixFactorizationRecommender
    },
    'dqn-mf': {
        'class': DqnInterviewer,
        'recommender': MatrixFactorizationRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-kg': {
        'class': DqnInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-collab': {
        'class': DqnInterviewer,
        'recommender': CollaborativePageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-joint': {
        'class': DqnInterviewer,
        'recommender': JointPageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'fmf': {
        'class': FMFInterviewer,
        'requires_interview_length': True,
    },
    'melu': {
        'class': MeLUInterviewer,
        'use_cuda': True
    }
}