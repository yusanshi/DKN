class Config():
    num_filters = 50
    window_sizes = [2, 3, 4]
    num_batches = 5000  # Number of batches to train
    num_batches_batch_loss = 50  # Number of batchs to show loss
    # Number of batchs to check loss and accuracy on test dataset
    num_batches_val_loss_and_acc = 500
    batch_size = 64
    learning_rate = 0.001
    train_split = 0.7
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    # If False, user_vector will be the average value of clicked_news_vector
    use_attention = True
    # If False, only word embedding and entity embedding will be used
    use_context = True

    num_words_a_sentence = 10
    num_word_tokens = 1 + 7947  # Don't modify this only if you use another dataset
    word_freq_threshold = 2
    entity_freq_threshold = 1
    word_embedding_dim = 50
    entity_embedding_dim = 50
