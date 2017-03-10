import tensorflow as tf
import random
import logging
import model_config as model_config
import numpy as np

class LstmSearchModel:
    def __init__(self,model_file=None,mode="query_processing",doc_vecs=None,config=None):
        #we provide 2 mode
        #training: making new model, query_processing: loading model from file and process the query
        #no support for incremental learning
        if mode=="training":
            self.is_training_mode=True
            self.config=model_config.ModelConfiguration()
        elif mode=="query_processing":
            if doc_vecs is None:
                raise Error("query_processing mode need index document vectors")
            else:
                self.doc_vecs=doc_vecs
            self.is_training_mode=False
            self.config=model_config.QueryProcessingModelConfiguration()
        else:
            raise Exception("mode must be query_processing or training")
        
        if config is not None:
            self.config=config
            
        self.vector_size=self.config.vector_size
        self.batch_size=self.config.batch_size
        self.max_term_seq=self.config.max_term_seq
        self.sess=tf.Session()
        
        if model_file is None:
            self.model_file=self.get_model_file_name()
        else:
            self.model_file=model_file
        
        if self.is_training_mode:
            logging.info(mode+" mode: initializing model...")
            self.create_lstm_model()
            self.lstm_init=True
        else: #query_processing mode
            logging.info(mode+" mode:loading model...")
            self.load_lstm_model()
            self.lstm_init=False
            
    
    def get_model_file_name(self):
        return self.config.lstm_model_dir+"/"+"lr_%f-vector_size_%d" %(self.config.lr,self.config.vector_size)+".lstm"
    
    def load_lstm_model(self):
        saver=tf.train.Saver()
        if self.model_file is None:
            self.model_file=self.get_model_file_name()
        saver.restore(self.sess,self.model_file)
        with tf.variable_scope(self.config.scope_name,reuse=True) as scope:
            self.define_training_ops(scope)
        
        
    def create_lstm_model(self):
        with tf.variable_scope(self.config.scope_name) as scope:
            self.define_training_ops(scope)
        init_op = tf.global_variables_initializer()
        logging.debug("initialization session start")
        self.sess.run(init_op)
        logging.debug("initialization session end")
    
    def define_training_ops(self,scope):
        logging.debug("defining variables")
        
        if self.is_training_mode:
            lstm_batch_size=self.batch_size*2
            self.labels=tf.placeholder("float",[self.batch_size,2])
        else:
            lstm_batch_size=self.batch_size
        
        #for feeding document or query
        #feed will be like [sentence_id, term_id, sentence_embedding_index]
        self.sentence_inputs=tf.placeholder("float",[lstm_batch_size,self.max_term_seq,self.vector_size])
        
        
        #for feeding each sentence length to stop lstm at the end of sentence 
        self.early_stop=tf.placeholder(tf.int32,[lstm_batch_size])
        
        #for feeding dropout late
        self.keep_prob=tf.placeholder(tf.float32)
        
        #define lstm
        lstm_bcell=tf.contrib.rnn.BasicLSTMCell(self.vector_size,lstm_batch_size)
        stm_bcell_drop=tf.contrib.rnn.DropoutWrapper(lstm_bcell, output_keep_prob=self.keep_prob)

        #lstm last output will be feeded to this regression node
        self.w_h=tf.get_variable("weight",[self.vector_size*2,2],initializer=tf.random_normal_initializer())
        self.b_h=tf.get_variable("biase",[2],initializer=tf.random_normal_initializer())
        
        self.initial_state=lstm_bcell.zero_state(lstm_batch_size, tf.float32)
        
        logging.debug("defining feedforward process")
        
        #calc lstm result
        logging.debug("defining lstm process")
        if not self.is_training_mode:
            scope.reuse_variables()
        self.converted_input=self.convert_to_lstm_input(self.sentence_inputs,lstm_batch_size)
        self.result,state=tf.contrib.rnn.static_rnn(
            lstm_bcell_drop
            ,self.convert_to_lstm_input(self.sentence_inputs,lstm_batch_size)
            ,initial_state=self.initial_state
            #,sequence_length=self.early_stop
            )
        
        self.sentence_outs=self.get_early_stop_outputs(self.result,self.early_stop,lstm_batch_size)
        
        
        if self.is_training_mode:
            self.querys_out,self.doc_outs=tf.split(self.sentence_outs,[self.batch_size,self.batch_size],0)
            self.concatinated=tf.concat([self.querys_out,self.doc_outs], 1)
            
            #translate query lstm result into document embedding space
            self.matmal=tf.matmul(self.concatinated,self.w_h)+self.b_h
            self.prediction=tf.nn.softmax(self.matmal)
        
            logging.debug("defining feed back phase")
        
            #learning
            self.loss=tf.reduce_mean(-tf.reduce_sum(self.labels*tf.log(self.prediction),reduction_indices=[1]))
            logging.debug("defining optimization process")
            self.train_step=tf.train.GradientDescentOptimizer(self.config.lr).minimize(self.loss)
        
        else: #query_processing mode
            #self.sentence_outs=tf.reshape(self.sentence_outs,[self.vector_size])
            doc_size=len(self.doc_vecs)
            self.query_outs=tf.concat([ self.sentence_outs for _ in range(doc_size)],0)
            self.doc_outs=tf.placeholder("float",[doc_size,self.vector_size])
            concatinated=tf.concat([self.query_outs,self.doc_outs], 1)
            self.prediction=tf.nn.softmax(tf.matmul(concatinated,self.w_h)+self.b_h)
        
        logging.debug("defining ops complete")
        
        return
    
    #gather lstm output
    def get_early_stop_outputs(self,outputs,early_stop,batch_size):
        index = tf.range(0, batch_size) * self.max_term_seq + (early_stop - 1)
        e_stop_outs = tf.gather(tf.reshape(outputs, [-1, self.vector_size]), index)
        return e_stop_outs
    
    #convert the feed input for lstm input
    #https://gist.github.com/evanthebouncy/8e16148687e807a46e3f
    def convert_to_lstm_input(self,feed_input,batch_size):
        seq_input=tf.transpose(feed_input,[1,0,2])
        return [tf.reshape(i, (batch_size, self.vector_size)) for i in tf.split(seq_input,self.max_term_seq,0)]
        

    # x: query vector seq
    def get_minibatch(self):
        x=[]
        y=[]
        result=[]
        for _ in range(self.batch_size):
            random_index=random.randint(0,len(self.querys)-1)
            x.append(self.querys[random_index])
            
            if bool(random.getrandbits(1)):
                y.append(self.documents[random_index])
                result.append([1.0,0])
            else:
                random_index=(random_index+random.randint(1,len(self.querys)-1))%len(self.documents)
                y.append(self.documents[random_index])
                result.append([0.0,1.0])
        x.extend(y)
        return x,result
    
    def set_training_data(self,querys, documents):
        assert len(querys)==len(documents)
        self.querys=querys
        self.documents=documents
        
    def train(self,epock):
        logging.info("training start; batch size:%d epock:%d",self.batch_size,epock)
        for i in range(epock):
            sentence_vec_seqs,match_result=self.get_minibatch()
            sentence_term_seq_lens=[len(vectors) for vectors in sentence_vec_seqs]
            
            feed={self.sentence_inputs: self.make_static_len_input(sentence_vec_seqs,self.max_term_seq)
                  ,self.early_stop: self.limit_term_seq_lens(sentence_term_seq_lens)
                  ,self.labels: match_result
                  ,self.keep_prob: self.config.keep_prob
                 }
            #print self.make_static_len_input(query_vec_seqs,self.max_term_seq)
            result=self.sess.run([self.train_step,
                                  self.loss,
                                  self.prediction,
                                  self.sentence_inputs,
                                  self.labels,
                                  self.w_h,
                                  self.matmal,
                                  self.sentence_outs,
                                  self.result,
                                  self.early_stop,
                                  self.converted_input,
                                  self.concatinated,
                                  self.initial_state],feed_dict=feed)
            logging.debug("epock%d loss %f",i,result[1])
            if i%10==0:
                logging.info("epock%d loss %f",i,result[1])
                logging.debug("initial state\n"+str(result[12]))
                logging.debug("inputs\n"+str(result[3]))
                logging.debug("early_stop\n"+str(result[9]))
                logging.debug("labels\n"+str(result[4]))
                logging.debug("converted input\n"+str(result[10]))
                logging.debug("lstm_result\n"+str(result[8]))
                logging.debug("sentence_outs\n"+str(result[7]))
                logging.debug("concatinated\n"+str(result[11]))
                logging.debug("w\n "+str(result[5]))
                logging.debug("matmal\n"+str(result[6]))
                logging.debug("prediction\n"+str(result[2]))
            
        self.save_model()
    
    def limit_term_seq_lens(self,term_seq_lens):
        for i in range(len(term_seq_lens)):
            if term_seq_lens[i] > self.max_term_seq:
                term_seq_lens[i]=self.max_term_seq
        return term_seq_lens
    
    #make variable length input into static size input(slicing or padding)
    def make_static_len_input(self,input_data,output_len):
        static_len_input=[]
        padding_content=[0.0]*self.vector_size
        for sentence in input_data:
            if len(sentence)>output_len:
                static_len_input.append(sentence[:output_len])
            else:
                to_be_padded_len=output_len-len(sentence)
                padded_sentence=sentence[:]
                for i in range(to_be_padded_len):
                    padded_sentence.append(padding_content)
                static_len_input.append(padded_sentence)
        return static_len_input
    
    def save_model(self):
        print(self.model_file)
        tf.train.Saver().save(self.sess,self.model_file)
        logging.info("model saved: "+self.model_file)
    
    def close_session(self):
        self.sess.close()
        logging.info("session closed")
    
    def get_matching_vector(self,query_w2v_seq):
        
        feed={self.sentence_inputs: self.make_static_len_input([query_w2v_seq],self.max_term_seq)
                  ,self.early_stop: self.limit_term_seq_lens([len(query_w2v_seq)])
                  ,self.doc_outs: self.doc_vecs
                  ,self.keep_prob: 1.0 #this is not train
                 }
        #print self.make_static_len_input(query_vec_seqs,self.max_term_seq)
        result=self.sess.run(self.prediction,feed_dict=feed)
        return result
        
        
    #return all documents vectors which wre used in training 
    def get_doc_vectors(self):
        lstm_batch_size=self.batch_size*2
        #reuse training node to output document vectors
        #we have to divide into batch_size to feed to the LSTM
        remain_docs=len(self.documents)
        result_vectors=[]
        padding_content=[[[0.0]*self.vector_size]*self.max_term_seq]
        while remain_docs > 0:
            if remain_docs > lstm_batch_size:
                input_batch_doc=self.documents[-remain_docs:-remain_docs+lstm_batch_size]
            else:
                #final chank
                input_batch_doc=self.documents[-remain_docs:]
                input_batch_doc.extend(padding_content*(lstm_batch_size-remain_docs)) #padding
                
            doc_term_seq_lens=[len(v) for v in input_batch_doc]
            
            feed={self.sentence_inputs: self.make_static_len_input(input_batch_doc,self.max_term_seq)
                  ,self.early_stop: self.limit_term_seq_lens(doc_term_seq_lens)
                  ,self.keep_prob: 1.0 #this is not train
                 }
            
            
            #print self.make_static_len_input(query_vec_seqs,self.max_term_seq)
            result=self.sess.run(self.sentence_outs,feed_dict=feed)
            
            if remain_docs < lstm_batch_size:
                result=result[:remain_docs] # remove padding 
                
            result_vectors.extend(list(result))
            remain_docs-=lstm_batch_size
            
        return result_vectors

