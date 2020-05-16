import os


class Config():
    num_filters = 50
    window_sizes = [2, 3, 4]
    num_batches = 8000  # Number of batches to train
    num_batches_batch_loss = 50  # Number of batchs to show loss
    # Number of batchs to check loss and accuracy on validation dataset
    num_batches_val_loss_and_acc = 300
    num_batches_save_checkpoint = 400
    batch_size = 256
    learning_rate = 0.001
    train_validation_split = (0.8, 0.2)
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    # TODO currently context embedding is not available
    # If False, only word embedding and entity embedding will be used
    use_context = os.environ[
        'CONTEXT'] == '1' if 'CONTEXT' in os.environ else False
    # If False, user_vector will be the average value of clicked_news_vector
    use_attention = os.environ[
        'ATTENTION'] == '1' if 'ATTENTION' in os.environ else True
    # Whether try to load checkpoint
    load_checkpoint = os.environ[
        'LOAD_CHECKPOINT'] == '1' if 'LOAD_CHECKPOINT' in os.environ else True
    num_words_a_news = 20
    entity_confidence_threshold = 0.5

    word_freq_threshold = 3
    entity_freq_threshold = 3

    # Modify this by the output of `src/dataprocess.py`
    num_word_tokens = 1 + 14760
    # Modify the following only if you use another dataset
    word_embedding_dim = 100
    entity_embedding_dim = 100
