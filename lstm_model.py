import tensorflow as tf
import random
import logging
import model_config as model_config
import numpy as np

class LstmSearchModel:
    def __init__(self,doc_size,model_file=None,mode="query_processing",doc_vecs=None,config=None,scope_reuse=None):
        self.doc_size=doc_size
        
        #when loading from same python kernel, reuse should be True 
        self.scope_reuse=scope_reuse
        
        #we provide 3 mode
        #training: making new model, 
        #resume_training: incremental learning 
        #query_processing: loading model from file and process the query
        if mode=="training" or mode=="resume_training":
            self.is_training_mode=True
            self.config=model_config.ModelConfiguration()
            
        elif mode=="query_processing":
            if doc_vecs is None:
                raise Error("query_processing mode need index document vectors")
            else:
                assert doc_size==len(doc_vecs)
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
        
        if mode=="training":
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
        with tf.variable_scope(self.config.scope_name,reuse=self.scope_reuse) as scope:
            self.define_ops(scope)
        saver=tf.train.Saver()
        if self.model_file is None:
            self.model_file=self.get_model_file_name()
        saver.restore(self.sess,self.model_file)
        
        
    def create_lstm_model(self):
        with tf.variable_scope(self.config.scope_name) as scope:
            self.define_ops(scope)
        init_op = tf.global_variables_initializer()
        logging.info("initialization session start")
        self.sess.run(init_op)
        logging.info("initialization session end")
    
    def define_ops(self,scope):
        logging.debug("defining variables")
        
        if self.is_training_mode:
            self.labels=tf.placeholder("float",[self.batch_size,self.doc_size])
        
        #for feeding document or query
        #feed will be like [sentence_id, term_id, sentence_embedding_index]
        self.query_inputs=tf.placeholder("float",[self.batch_size,self.max_term_seq,self.vector_size])
        self.doc_inputs=tf.placeholder("float",[self.doc_size,self.max_term_seq,self.vector_size])
        
        
        #for feeding each sentence length to stop lstm at the end of sentence
        self.early_stop_query=tf.placeholder(tf.int32,[self.batch_size]) 
        self.early_stop_doc=tf.placeholder(tf.int32,[self.doc_size])
        
        #for feeding dropout late
        self.keep_prob=tf.placeholder(tf.float32)
        
        #define lstm
        lstm_bcell=tf.contrib.rnn.BasicLSTMCell(self.vector_size,self.doc_size)
        lstm_bcell_drop=tf.contrib.rnn.DropoutWrapper(lstm_bcell, output_keep_prob=self.keep_prob)

        #lstm last output will be feeded to this regression node
        self.w_q=tf.get_variable("w_q",[self.vector_size,self.vector_size],initializer=tf.random_normal_initializer())
        self.b_q=tf.get_variable("b_q",[self.vector_size],initializer=tf.random_normal_initializer())
        self.w_d=tf.get_variable("w_d",[self.vector_size,self.vector_size],initializer=tf.random_normal_initializer())
        self.b_d=tf.get_variable("b_d",[self.vector_size],initializer=tf.random_normal_initializer())
        
        logging.debug("defining feedforward process")
        
        #calc lstm result
        logging.debug("defining lstm process")
#         if not self.is_training_mode and self.scope_reuse:
#             scope.reuse_variables()
        
        self.results_query,state=tf.contrib.rnn.static_rnn(
            lstm_bcell_drop
            ,self.convert_to_lstm_input(self.query_inputs,self.batch_size)
            ,sequence_length=self.early_stop_query
            ,dtype=tf.float32
            )
        self.query_outs=self.get_early_stop_outputs(self.results_query,self.early_stop_query)
            
        if self.is_training_mode:
            scope.reuse_variables()
            self.results_doc,state=tf.contrib.rnn.static_rnn(
                lstm_bcell_drop
                ,self.convert_to_lstm_input(self.doc_inputs,self.doc_size)
                ,sequence_length=self.early_stop_doc
                ,dtype=tf.float32
                )

            self.doc_outs=self.get_early_stop_outputs(self.results_doc,self.early_stop_doc)
        else:
            self.doc_outs=tf.placeholder("float",[self.doc_size,self.vector_size])
        
        #hidden layer
        self.query_hidden=tf.matmul(self.query_outs,self.w_q)+self.b_q
        self.doc_hidden=tf.matmul(self.doc_outs,self.w_d)+self.b_d
        
        #calculating cosine similarity
        self.query_hidden_norm=self.normarize(self.query_hidden)
        self.doc_hidden_norm=self.normarize(self.doc_hidden)
        self.matmul=tf.transpose(tf.matmul(self.doc_outs,self.query_outs,transpose_b=True))
        self.prediction=tf.nn.softmax(self.matmul)
        
        if self.is_training_mode:
            self.dynamic_lr=tf.placeholder(tf.float32)
            
            #learning
            self.loss=tf.reduce_mean(-tf.reduce_sum(self.labels*tf.log(self.prediction),reduction_indices=[1]))
            L2_sqr=tf.nn.l2_loss(self.w_d)+tf.nn.l2_loss(self.w_q)
            self.cost=self.loss+L2_sqr*self.config.lambda2
            logging.debug("defining optimization process")
            self.train_step=tf.train.GradientDescentOptimizer(self.dynamic_lr).minimize(self.cost)
        
        logging.debug("defining ops complete")
        
        return
    
    def normarize(self,a):
        a_magnitude=tf.sqrt(tf.reduce_sum(tf.multiply(a,a),axis=1))
        a_length=a.shape.dims[0].value
        return tf.div(a,tf.reshape(a_magnitude,[a_length,1]))
    
    #gather lstm output
    def get_early_stop_outputs(self,outputs,early_stop):
        batch_size=outputs[0].shape.dims[0].value
        flat=tf.reshape(outputs, [-1, self.vector_size])
        
        index = tf.range(0, batch_size)+(early_stop - 1)*batch_size
        e_stop_outs = tf.gather(flat, index)
        return e_stop_outs
    
    #convert the feed input for lstm input
    #https://gist.github.com/evanthebouncy/8e16148687e807a46e3f
    def convert_to_lstm_input(self,feed_input,batch_size):
        seq_input=tf.transpose(feed_input,[1,0,2])
        return [tf.reshape(i, (batch_size, self.vector_size)) for i in tf.split(seq_input,self.max_term_seq,0)]
        

    # x: query vector seq
    def get_minibatch(self):
        querys=[]
        labels=[]
        for _ in range(self.batch_size):
            random_index=random.randint(0,len(self.querys)-1)
            querys.append(self.querys[random_index])
            label=[0]*len(self.querys)
            label[random_index]=1 #one hot vector
            labels.append(label)
        return querys,labels
    
    def set_training_data(self,querys, documents):
        assert len(querys)==len(documents)
        self.querys=querys
        self.documents=documents
        
    def train(self,epock):
        lr=self.config.lr
        loss_sum=0
        prev_loss_sum=10000000
        
        logging.info("training start; batch size:%d epock:%d learning rate:%f",self.batch_size,epock,lr)
        for i in range(epock):
            query_vec_seq,labels=self.get_minibatch()
            query_term_seq_lens=[len(vectors) for vectors in query_vec_seq]
            doc_term_seq_lens=[len(vectors) for vectors in self.documents]
            
            feed={self.query_inputs:self.make_static_len_input(query_vec_seq, self.max_term_seq)
                  ,self.doc_inputs: self.make_static_len_input(self.documents,self.max_term_seq)
                  ,self.early_stop_query: self.limit_term_seq_lens(query_term_seq_lens)
                  ,self.early_stop_doc: self.limit_term_seq_lens(doc_term_seq_lens)
                  ,self.labels: labels
                  ,self.keep_prob: self.config.keep_prob
                  ,self.dynamic_lr:lr
                 }
            #print self.make_static_len_input(query_vec_seqs,self.max_term_seq)
            result=self.sess.run([self.train_step,self.loss],feed_dict=feed,)
            logging.debug("epock%d loss %f",i,result[1])
            loss_sum+=result[1]
            if i%10==0:
                if i%50==0:
                    self.save_model()
                logging.info("epock%d loss %f",i,result[1])

                if loss_sum > prev_loss_sum:
                    #lr/=2.0
                    logging.info("learning rate change: %f",lr)
                prev_loss_sum=loss_sum
                loss_sum=0
                
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
        tf.train.Saver().save(self.sess,self.model_file)
        logging.info("model saved: "+self.model_file)
    
    def close_session(self):
        self.sess.close()
        logging.info("session closed")
    
    def get_matching_vector(self,query_w2v_seq):
        
        feed={self.query_inputs: self.make_static_len_input([query_w2v_seq],self.max_term_seq)
              ,self.early_stop_query: self.limit_term_seq_lens([len(query_w2v_seq)])
              ,self.doc_outs: self.doc_vecs
              ,self.keep_prob: 1.0 #this is not train
             }
        #print self.make_static_len_input(query_vec_seqs,self.max_term_seq)
        result=self.sess.run(self.prediction,feed_dict=feed)
        return result
        
        
    #return all documents vectors which wre used in training 
    def get_doc_vectors(self):
        doc_term_seq_lens=[len(v) for v in self.documents]
        feed={self.doc_inputs: self.make_static_len_input(self.documents,self.max_term_seq)
              ,self.early_stop_doc: self.limit_term_seq_lens(doc_term_seq_lens)
              ,self.keep_prob: 1.0 #this is not train
             }
        
        result_vectors=self.sess.run(self.doc_outs,feed_dict=feed)
        
        return result_vectors

