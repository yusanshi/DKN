class Config():
    num_filters = 50
    window_sizes = [2, 3, 4]
    num_batches = 5000  # Number of batches to train
    num_batches_batch_loss = 40  # Number of batchs to show loss
    num_batches_val_loss = 500  # Number of batchs to check loss on test dataset
    batch_size = 64
    learning_rate = 1.0
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    # If False, user_vector will be the average value of clicked_news_vector
    use_attention = True
    # If False, only word embedding and entity embedding will be used
    use_context = True

    # Determined by knowledge graph embedding training
    # If you want to modify these, remember to also adjust parameters
    # for your KG traning (Tips: in news_preprocess.py and kg_preprocess.py,
    # See README.md for more details)
    num_words_a_sentence = 10
    word_embedding_dim = 50
    entity_embedding_dim = 50
