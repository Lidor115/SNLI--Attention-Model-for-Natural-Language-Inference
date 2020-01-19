from Parser import Parser

if __name__ == '__main__':
    train_parser = Parser('./Data/snli_1.0_train.jsonl')
    train_parser.DataLoader(100)
    dev_parser = Parser('./Data/snli_1.0_dev.jsonl')
    test_parser = Parser('./Data/snli_1.0_test.jsonl')
    embedding_dim = 300
    hidden_dim_1 = 1000
    batch_size = 20
    output_dim = 3
    #todo - MODEL
    #  - Iterate model
    # - predict

