import model_config
import lstm_model
import logging

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s %(levelname)-7s %(funcName)s %(lineno)d %(message)s")


def save_vectors(filename,vectors):
    with open(filename,"w") as out:
        for vector in vectors:
            line=""
            for val in vector:
                line+=str(val)+" "  
            out.write(line+"\n")

def load_vectors(filename):
    with open(filename,"r") as f:
        vectors=[]
        for line in f.readlines():
            vector=[]
            split=line.rstrip().split(" ")
            for val in split:
                vector.append(float(val))
            vectors.append(vector)
    return vectors

class ModelTestingConfig(model_config.ModelConfiguration):
    def __init__(self):
        super().__init__()
        self.lr=1
        self.vector_size=3
        self.keep_prob=0.5
        self.batch_size=10
        self.lstm_model_dir="model/lstm_model"
        self.scope_name="lstm_search"
        self.max_term_seq=5

class TestModelTestingConfig(ModelTestingConfig):
    def __init__(self):
        super().__init__()
        self.batch_size=1

        

querys=[[[1,0,0]],[[0,1,0]],[[0,0,1]]]
documents=[[[0,0,0],[0,0,0],[1,0,0]],[[0,0,0],[0,1,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,1]]]
#documents=[[[1,0,0]],[[0,1,0]],[[0,0,1]]]

#making index
doc_size=len(documents)
training_model=lstm_model.LstmSearchModel(doc_size,mode="training",config=ModelTestingConfig())
training_model.set_training_data(querys,documents)
training_model.train(1000)
training_model.close_session()

#close and resume training 
training_model=lstm_model.LstmSearchModel(doc_size,mode="resume_training",config=ModelTestingConfig())
training_model.set_training_data(querys,documents)
training_model.train(50)
doc_vecs=training_model.get_doc_vectors()
print(doc_vecs)
training_model.close_session()

#query processing
qp_model=lstm_model.LstmSearchModel(doc_size,mode="query_processing",doc_vecs=doc_vecs,config=TestModelTestingConfig())
print(qp_model.get_matching_vector([[1,0,0]]))
print(qp_model.get_matching_vector([[0,1,0]]))
print(qp_model.get_matching_vector([[0,0,1]]))

save_vectors("tmp.vec",doc_vecs)
doc_vecs_saved=load_vectors("tmp.vec")
qp_model=lstm_model.LstmSearchModel(doc_size,mode="query_processing",doc_vecs=doc_vecs_saved,config=TestModelTestingConfig())

print(qp_model.get_matching_vector([[1,0,0]]))
print(qp_model.get_matching_vector([[0,1,0]]))
print(qp_model.get_matching_vector([[0,0,1]]))
