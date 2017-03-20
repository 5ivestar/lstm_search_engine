import logging
import lstm_model

#wrapper class for lstm_search_model to manage learning, indexing and searching

class VectorSearchEngine:
    
    #mode is indexing or searching
    def __init__(self,data_handler,doc_vec_file_name=None,mode="searching"):
        self.data_handler=data_handler
        self.mode=mode
        self.doc_vec_file_name=doc_vec_file_name
        
        if mode!="indexing" and mode!="searching":
            raise Exception("mode should be indexing or searching: "+mode)
        
        self.initialize_model()
        
    def initialize_model(self):
        doc_size=self.data_handler.get_data_num()
        
        if self.doc_vec_file_name is None:
            self.doc_vec_file_name=self.get_doc_vecs_file_name()
        
        if self.mode=="searching":
            self.load_doc_vectors()
            self.lstm_model=lstm_model.LstmSearchModel(doc_size,mode="query_processing",doc_vecs=self.docvectors,scope_reuse=None)
            
        else: #indexing
            self.lstm_model=lstm_model.LstmSearchModel(doc_size,mode="training")
     
            
    def get_doc_vecs_file_name(self):
        return "index.docvecs"
    
    def learning_and_indexing(self):
        if self.lstm_model is None:
            raise Exception("model isn't initialized")
        
        querys,documents,self.urls=self.data_handler.make_training_data()
        self.lstm_model.set_training_data(querys,documents)
        self.lstm_model.train(50)
 
        self.docvectors=self.lstm_model.get_doc_vectors()
        self.save_doc_vectors(self.doc_vec_file_name)
    
        
    def load_doc_vectors(self):
        with open(self.doc_vec_file_name,"r") as f:
            self.docvectors=[]
            self.urls=[]
            for line in f.readlines():
                vector=[]
                split=line.rstrip().split(" ")
                self.urls.append(split[0])
                for val in split[1:]:
                    vector.append(float(val))
                self.docvectors.append(vector)
        logging.info("complelte to load document vectors from %s",self.doc_vec_file_name)
    #file format
    #id,url
    def save_doc_vectors(self,file_name):
        assert len(self.docvectors)==len(self.urls), str(len(self.docvectors))+" "+str(self.urls)
        with open(file_name,"w") as f:
            for vec,url in zip(self.lstm_model.get_doc_vectors(),self.urls):
                line=url+" "
                for val in vec:
                    line+=str(val)+" "
                f.write(line+"\n")
        logging.info("saveing docvec file "+file_name)
    
    def search(self,query,result_size=1):
        if self.lstm_model is None:
            raise Excepion("model isn't initialized")
        query_vec_seq=self.data_handler.sentence_to_seqw2v(query,{})
        matching_vec=self.lstm_model.get_matching_vector(query_vec_seq)[0].tolist()
        
        merge_queue=[ [i,val] for i,val in enumerate(matching_vec)]
        id=sorted(merge_queue,key=lambda x:x[1],reverse=True)
        return self.urls[id[0][0]],self.urls[id[1][0]]
    
    def close_engine(self):
        self.lstm_model.close_session()
    

class Document:
    def __init__(self,id,url,vec):
        self.url=url
        self.id=id
        self.vec=vec

# if __name__=="__main__":
    

