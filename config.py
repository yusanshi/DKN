class Config():
    filter_out_channels = 100
    window_sizes = [1, 2, 3, 4]  # TODO
    num_batches = 1000  # Number of batches to train
    num_batches_batch_loss = 20  # Number of batchs to show loss
    num_batches_val_loss = 100  # Number of batchs to check loss on test dataset
    batch_size = 128
    learning_rate = 0.001
    num_workers = 4  # Number of workers for data loading

    num_clicked_news_a_user = 50  # Number of sampled click history for each user

    # Determined by knowledge graph embedding training
    # If you want to modify these, remember to also adjust parameters
    # for your KG traning (Tips: in news_preprocess.py and kg_preprocess.py,
    # See README.md for more details)
    num_words_a_sentence = 10
    word_embedding_dim = 50
    entity_embedding_dim = 50
