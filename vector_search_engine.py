import logging

class VectorSearchEngine:
    def __init__(self,lstm_model,load_file_name=None):
        self.lstm_model=lstm_model
        
        if load_file_name is not None:
            self.load_doc_vectors(load_file_name)
            logging.info("compelte to load document vectors from %s",load_file_name)
        
    def load_doc_vectors(self,load_file_name):
        self.indexed_docs=[]
        with open(load_file_name) as f:
            for line in f.readlines():
                row=line.rstrip().split(" ")
                doc_vec=[float(val) for val in row[2:]]
                self.indexed_docs.append(Document(row[0],row[1],doc_vec))
        
        
    def add_document(doc_label,doc_vector):
    
    """
        file format
        id url vec
    """
    def save_doc_vectors(self,file_name,url_list):
        assert len(self.indexed_docs)==len(url_list)
        with open(file_name,"w") as f:
            for doc in self.indexed_docs:
                line=str(doc.id)+" "+doc.url
                for val in doc.vec:
                    line+=str(val)
                f.write(line+"\n")
    
    def search(query_vector):
        for indexed_doc in self.indexed_docs:
            self.calc_distance(query_vector,indexed_doc.vec)
            #TODO implement
        return top_result
    
    def calc_distance(self,query,doc):
        distance=0.0
        for x,y in zip(query,doc):
            distance+=(x-y)*(x-y)
        return distance

class Document:
    def __init__(self,id,url,vec):
        self.url=url
        self.id=id
        self.vec=vec
        
        