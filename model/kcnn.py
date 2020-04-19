import torch
import torch.nn as nn
import torch.nn.functional as F


class KCNN(torch.nn.Module):
    """
    Knowledge-aware CNN (KCNN) based on Kim CNN.
    Input a news sentence (e.g. its title), produce its embedding vector.
    """

    def __init__(self, config, embeddings):
        super(KCNN, self).__init__()
        self.config = config
        self.embeddings = embeddings
        self.transform_matrix = nn.Parameter(
            torch.empty(self.config.entity_embedding_dim,
                        self.config.word_embedding_dim))
        self.transform_bias = nn.Parameter(
            torch.empty(self.config.word_embedding_dim))

        self.conv_filters = nn.ModuleDict({
            str(x): nn.Conv2d(3, self.config.filter_out_channels,
                              (x, self.config.word_embedding_dim))
            for x in self.config.window_sizes
        })

        self.transform_matrix.data.uniform_(-0.1, 0.1)
        self.transform_bias.data.uniform_(-0.1, 0.1)

    def forward(self, news):
        """
        Args:
          news:
            {
                "word": [Tensor(batch_size) * num_words_a_sentence],
                "entity":[Tensor(batch_size) * num_words_a_sentence]
            }

        Returns:
          batch_size * (len(window_sizes) * filter_out_channels)
        """
        # batch_size, num_words_a_sentence, word_embedding_dim
        word_vector = self.embeddings["word"](news["word"])
        # batch_size, num_words_a_sentence, entity_embedding_dim
        entity_vector = self.embeddings["entity"](news["entity"])
        # batch_size, num_words_a_sentence, entity_embedding_dim
        context_vector = self.embeddings["context"](news["entity"])

        # TODO ei and ei2 are set as zero if wi has no corresponding entity

        # batch_size, num_words_a_sentence, word_embedding_dim
        transformed_entity_vector = torch.tanh(
            torch.matmul(entity_vector, self.transform_matrix) +
            self.transform_bias)
        # batch_size, num_words_a_sentence, word_embedding_dim
        transformed_context_vector = torch.tanh(
            torch.matmul(context_vector, self.transform_matrix) +
            self.transform_bias)
        # batch_size, 3, num_words_a_sentence, word_embedding_dim
        multi_channel_vector = torch.stack([
            word_vector, transformed_entity_vector, transformed_context_vector
        ], dim=1)

        pooled_vectors = []
        for x in self.config.window_sizes:
            # batch_size, filter_out_channels, num_words_a_sentence + 1 - x
            convoluted = self.conv_filters[str(x)](
                multi_channel_vector).squeeze(dim=3)
            # batch_size, filter_out_channels, num_words_a_sentence + 1 - x
            activated = F.relu(convoluted)
            # TODO: vs nn.MaxPool1d
            # batch_size, filter_out_channels
            pooled = activated.max(dim=-1)[0]
            pooled_vectors.append(pooled)
        # batch_size, len(window_sizes) * filter_out_channels
        final_vector = torch.cat(pooled_vectors, dim=1)
        return final_vector
