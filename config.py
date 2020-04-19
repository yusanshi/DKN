class Config():
    word_embedding_dim = 100
    entity_embedding_dim = 80  # TODO adjust to 100
    filter_out_channels = 120  # TODO adjust to 100
    window_sizes = [1, 2, 3, 4]  # TODO
    num_batches = 1000  # Number of batches to train
    num_batches_batch_loss = 20  # Number of batchs to show loss
    num_batches_val_loss = 100  # Number of batchs to check loss on test dataset
    batch_size = 64
    learning_rate = 0.00002
    num_workers = 4  # Number of workers for data loading

    # Determined by the dataset
    # Don't modify it if you use my dataset
    num_word_tokens = 8033
    num_entity_tokens = 3777
    num_words_a_sentence = 10
    num_clicked_news_a_user = 278
