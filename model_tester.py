import model_config
import lstm_model
import logging

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s %(levelname)-7s %(funcName)s %(lineno)d %(message)s")


class ModelTestingConfig(model_config.ModelConfiguration):
    def __init__(self):
        super().__init__()
        self.lr=1.0
        self.vector_size=3
        self.keep_prob=0.5
        self.batch_size=100
        self.lstm_model_dir="model/lstm_model"
        self.scope_name="lstm_search"
        self.max_term_seq=3

class TestModelTestingConfig(ModelTestingConfig):
    def __init__(self):
        super().__init__()
        self.batch_size=1

        

querys=[[[1,0,0]],[[0,1,0]],[[0,0,1]]]
documents=[[[0,0,0],[1,0,0],[0,0,0]],[[0,1,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,1],[0,0,0]]]

#making index
training_model=lstm_model.LstmSearchModel(mode="training",config=ModelTestingConfig())
training_model.set_training_data(querys,documents)
training_model.train(1000)
doc_vecs=training_model.get_doc_vectors()
print(doc_vecs)
training_model.close_session()

#query processing
qp_model=lstm_model.LstmSearchModel(mode="query_processing",doc_vecs=list(doc_vecs[0]),config=TestModelTestingConfig())
print(qp_model.get_matching_vector([[1,0,0]]))

