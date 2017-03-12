import csv
import logging

#private
import lstm_model as lstm_model
import w2v_handler as w2v_handler
import stemmer as stm
import data_handler as dh

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


logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)-7s %(funcName)s %(lineno)d %(message)s")

w2v_model_file_name="model/w2v_model/wiki_kc_mix"
#kc_data_file_name="data/kc_data_small.csv"
kc_data_file_name="data/kc_data.csv"

w2v_model=w2v_handler.load_word2vec_model(w2v_model_file_name,"","")
data_handler=dh.KcDataHandler(kc_data_file_name,stm.SimpleStemmer(),w2v_model)
querys,documents,vector_cache=data_handler.make_training_data()
print(len(querys),len(documents),list(vector_cache.keys())[:10])

doc_size=len(documents)
training_model=lstm_model.LstmSearchModel(doc_size,mode="training")
training_model.set_training_data(querys,documents)
training_model.train(500)
doc_vecs=training_model.get_doc_vectors()
training_model.close_session()
save_vectors("tmp.vecs", doc_vecs)

qp_model=lstm_model.LstmSearchModel(doc_size,mode="query_processing",doc_vecs=doc_vecs)
print(qp_model.get_matching_vector(querys[0]))
print(qp_model.get_matching_vector(querys[1]))
print(qp_model.get_matching_vector(querys[3]))




