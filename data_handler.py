import csv
import codecs
import logging

class KcDataHandler:
    def __init__(self,file_name,stemmer,w2v_model):
        self.file_name=file_name
        self.stemmer=stemmer
        self.w2v_model=w2v_model
        
    #vector_cache: cacheing word embedding vector as tensorflow static vector.
    #querys and documents reference vector_cache to build their vector sequence
    def make_training_data(self):
        vector_cache={}
        querys=[]
        documents=[]
        with codecs.open(self.file_name,encoding="utf-8") as f:
            reader=csv.reader(f)
            
            #skip header
            next(reader)
            
            for row in reader:
                #row[1]:page_title row[2]:section_title
                query=self.sentence_to_seqw2v(row[1]+" "+row[2],vector_cache)
                #row[3] content
                document=self.sentence_to_seqw2v(row[3],vector_cache)
                
                if len(query)>0 and len(document)>0:
                    querys.append(query)
                    documents.append(document) 
            
        return querys,documents,vector_cache
    
    
    def sentence_to_seqw2v(self,sentence,vector_cache):
        tokens=self.stemmer.get_stemed_tokens(sentence)
        vector_seq=[]
        for token in tokens:
            if not token in vector_cache.keys():
                try:
                    token_vec=self.w2v_model[token]
                    #token_tf_tensor=tf.convert_to_tensor([value for value in token_vec])
                    #vector_seq.append(token_tf_tensor)
                    #vector_cache[token]=token_tf_tensor
                    token_vec=[value for value in token_vec]
                    vector_seq.append(token_vec)
                    vector_cache[token]=token_vec
                except KeyError:
                    #if the token didn't registered in w2v model, ignore
                    logging.warning('"%s" cannot be vectorized',token)
                    continue
            else:
                vector_seq.append(vector_cache[token])
        
        return vector_seq
