import csv
import logging

#private
import w2v_handler as w2v_handler
import stemmer as stm
import data_handler as dh
import vector_search_engine as vse

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s %(levelname)-7s %(funcName)s %(lineno)d %(message)s")

w2v_model_file_name="model/w2v_model/wiki_kc_mix"
kc_data_file_name="data/kc_data_small.csv"
# kc_data_file_name="data/kc_data.csv"

#load kc data with word2vec
w2v_model=w2v_handler.load_word2vec_model(w2v_model_file_name,"","")
data_handler=dh.KcDataHandler(kc_data_file_name,stm.SimpleStemmer(),w2v_model)

engine=vse.VectorSearchEngine(data_handler,mode="indexing")
engine.learning_and_indexing()
engine.close_engine()




