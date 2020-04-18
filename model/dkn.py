import torch
import torch.nn as nn
from model.kcnn import KCNN
from model.attention import Attention
from model.dnn_g import DNN_G


class DKN(torch.nn.Module):
    """
    Deep knowledge-aware network.
    Input a candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self):
        super(DKN, self).__init__()
        self.kcnn = KCNN()
        self.attention = Attention()
        self.dnn_g = DNN_G()

    def forward(self, candidate_news, clicked_news):
        candidate_news_vector = self.kcnn(candidate_news)
        clicked_news_vector = self.kcnn(clicked_news)
        user_vector = self.attention(
            candidate_news_vector, clicked_news_vector)
        click_probability = self.dnn_g(candidate_news_vector, user_vector)
        return click_probability
