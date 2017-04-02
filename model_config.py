class ModelConfiguration:
    def __init__(self):
        self.lr=0.1
        self.vector_size=200
        self.keep_prob=1
        self.batch_size=100
        self.lstm_model_dir="model/lstm_model"
        self.scope_name="lstm_search"
        self.max_term_seq=500
        self.lambda2=0.01
        
class QueryProcessingModelConfiguration(ModelConfiguration):
    def __init__(self):
        super().__init__()
        self.max_term_seq=10
        self.batch_size=1
        self.keep_prob=1
