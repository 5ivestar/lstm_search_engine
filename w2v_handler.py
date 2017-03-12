from gensim.models import word2vec 
import os

def load_word2vec_model(model_file,corpus_file,vector_size=200):
    if os.path.exists(model_file):
        model= word2vec.Word2Vec.load(model_file)
    elif os.path.exists(corpus_file):
        data =word2vec.Text8Corpus(corpus_file)
        model=word2vec.Word2Vec(data,size=vector_size)
        model.save(model_file)
    
    else:
        raise Exception("neither model_file nor corpus file doesn't exists")
    
    return model

if __name__=="__main__":
    corpus_file="data/wiki_2015_knowledgecenter.txt"
    model_file="wiki_kc_mix"
    load_word2vec_model(model_file,corpus_file)