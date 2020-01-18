from Parser import Parser

if __name__ == '__main__':
    train_parser = Parser('./Data/snli_1.0_train.jsonl')
    data_train, max_len_train = train_parser.Parse()
    dev_parser = Parser('./Data/snli_1.0_dev.jsonl')
    data_dev, max_len_dev = dev_parser.Parse()
    test_parser = Parser('./Data/snli_1.0_test.jsonl')
    test_train, max_len_test = test_parser.Parse()
    embedding_dim = 300
    hidden_dim_1 = 1000
    batch_size = 20
    output_dim = 3
    #todo - MODEL
    #todo  - Iterate model
    #todo - predict

