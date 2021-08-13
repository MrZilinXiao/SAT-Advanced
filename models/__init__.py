from termcolor import colored

# from models.backbone.dgcnn import DGCNN
from models.backbone.lstm_encoder import LSTMEncoder
from models.backbone.mlp import MLP
from models.backbone.word_embeddings import load_glove_pretrained_embedding, make_pretrained_embedding

try:
    from models.backbone.point_net_pp import PointNetPP
except ImportError:
    PointNetPP = None
    msg = colored('Pnet++ is not found. Hence you cannot run all models. Install it via '
                  'external_tools (see README.txt there).', 'red')
    print(msg)
