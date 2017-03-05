import csv
import logging

#private
import lstm_model as lstm_model
import w2v_handler as w2v_handler
import stemmer as stm
import data_handler as dh

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s %(levelname)-7s %(funcName)s %(lineno)d %(message)s")

w2v_model_file_name="w2v_wiki.mod"
kc_data_file_name="kc_data_small.csv"

w2v_model=w2v_handler.load_word2vec_model(w2v_model_file_name,"","")
data_handler=dh.KcDataHandler(kc_data_file_name,stm.SimpleStemmer(),w2v_model)
querys,documents,vector_cache=data_handler.make_training_data()
print(len(querys),len(documents),list(vector_cache.keys())[:10])

training_model=lstm_model.LstmSearchModel(mode="training")
training_model.set_training_data(querys,documents)
training_model.train(100)
print(training_model.get_doc_vectors())
training_model.close_session()

restored=lstm_model.LstmSearchModel(mode="query_processing")
print(restored.get_query_vector(querys[0]))




