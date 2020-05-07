from models.ddpg.ddpg_interviewer import DDPGInterviewer
from models.dqn.dqn_interviewer import DqnInterviewer
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
from recommenders.pagerank.linear_collaborative_pagerank_recommender import LinearCollaborativePageRankRecommender
from recommenders.pagerank.linear_joint_pagerank_recommender import LinearJointPageRankRecommender
from recommenders.pagerank.linear_kg_pagerank_recommender import LinearKGPageRankRecommender
from recommenders.random.random_recommender import RandomRecommender
from recommenders.remote.particle_filtering_recommender import ParticleFilteringRecommender
from recommenders.svd.svd_recommender import SVDRecommender
from recommenders.toppop.toppop_recommender import TopPopRecommender

models = {
    'ddpg': {
        'class': DDPGInterviewer,
        'recommender': JointPageRankRecommender
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
    'greedy-ppr-collab': {
        'class': GreedyInterviewer,
        'recommender': CollaborativePageRankRecommender
    },
    'greedy-ppr-joint': {
        'class': GreedyInterviewer,
        'recommender': JointPageRankRecommender
    },
    'greedy-ppr-kg-rec': {
        'class': GreedyInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender,
        'interviewer_kwargs': {
            'recommendable_only': True
        },
        'recommender_kwargs': {
            'recommendable_only': True
        }
    },
    'greedy-ppr-collab-rec': {
        'class': GreedyInterviewer,
        'recommender': CollaborativePageRankRecommender,
        'interviewer_kwargs': {
            'recommendable_only': True
        },
        'recommender_kwargs': {
            'recommendable_only': True
        }
    },
    'greedy-ppr-joint-rec': {
        'class': GreedyInterviewer,
        'recommender': JointPageRankRecommender,
        'interviewer_kwargs': {
            'recommendable_only': True
        },
        'recommender_kwargs': {
            'recommendable_only': True
        }
    },
    'lrmf': {
        'class': LRMFInterviewer,
        'requires_interview_length': True,
        'use_cuda': False
    },
    'naive-pf': {
        'class': NaiveInterviewer,
        'recommender': ParticleFilteringRecommender
    },
    'naive-svd': {
        'class': NaiveInterviewer,
        'recommender': SVDRecommender
    },
    'naive-ppr-collab': {
        'class': NaiveInterviewer,
        'recommender': CollaborativePageRankRecommender
    },
    'naive-ppr-kg': {
        'class': NaiveInterviewer,
        'recommender': KnowledgeGraphPageRankRecommender
    },
    'naive-ppr-joint': {
        'class': NaiveInterviewer,
        'recommender': JointPageRankRecommender
    },
    'naive-ppr-linear-collab': {
        'class': NaiveInterviewer,
        'recommender': LinearCollaborativePageRankRecommender
    },
    'naive-ppr-linear-joint': {
        'class': NaiveInterviewer,
        'recommender': LinearJointPageRankRecommender
    },
    'naive-ppr-linear-kg': {
        'class': NaiveInterviewer,
        'recommender': LinearKGPageRankRecommender
    },
    'naive-knn': {
      'class': NaiveInterviewer,
      'recommender': KNNRecommender
    },
    'dqn-knn': {
        'class': DqnInterviewer,
        'recommender': KNNRecommender,
        'requires_interview_length': True
    },
    'naive-mf': {
        'class': NaiveInterviewer,
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
    'dqn-ppr-linear-kg': {
        'class': DqnInterviewer,
        'recommender': LinearKGPageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-linear-joint': {
        'class': DqnInterviewer,
        'recommender': LinearJointPageRankRecommender,
        'requires_interview_length': True,
        'use_cuda': True
    },
    'dqn-ppr-linear-collab': {
        'class': DqnInterviewer,
        'recommender': LinearCollaborativePageRankRecommender,
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