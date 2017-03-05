from gensim.models import word2vec 
import os

def load_word2vec_model(model_file,corpus_file,vector_size=200):
    if os.path.exists(model_file):
        model= word2vec.Word2Vec.load(model_file)
    else:
        data =word2vec.Text8Corpus(corpus_file)
        model=word2vec.Word2Vec(data,size=vector_size)
        model.save(model_file)
    
    return model