class ModelConfiguration:
    def __init__(self):
        self.lr=0.001
        self.vector_size=200
        self.keep_prob=0.5
        self.batch_size=100
        self.lstm_model_dir="model/lstm_model"
        self.scope_name="lstm_search"
        self.max_term_seq=3
        
class QueryProcessingModelConfiguration(ModelConfiguration):
    def __init__(self):
        super().__init__()
        self.max_term_seq=100
        self.batch_size=1