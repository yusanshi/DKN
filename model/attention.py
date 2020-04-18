import torch
from model.dnn_h import DNN_H


class Attention(torch.nn.Module):
    """
    Attention Net.
    Input embedding vectors (produced by KCNN) of a candidate news and all of user's clicked news,
    produce final user embedding vectors with respect to the candidate news.
    """

    def __init__(self):
        super(Attention, self).__init__()
        self.dnn_h = DNN_H()

    def forward(self, candidate_news_vector, clicked_news_vector):
        user_vector = torch.ones(4, 5)

        return user_vector
