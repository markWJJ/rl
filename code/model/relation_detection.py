import tensorflow as tf
import numpy as np
import os

def _build_bi_lstm(inputs, input_lens, hidden_size, initial_state_fw=None, initial_state_bw=None, scope=None):
    
    f_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    b_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=f_cell, 
                                        cell_bw=b_cell,
                                        initial_state_fw=initial_state_fw,
                                        initial_state_bw=initial_state_bw,
                                        dtype="float",
                                        sequence_length=input_lens,
                                        inputs=inputs, 
                                        scope=scope)
    
    return tf.concat([outputs[0], outputs[1]], axis=-1), states
    
class RelationDetection(object):
    def __init__(self, params, sess):
        self.hidden_size = params['hidden_size']
        self.q_tokens = tf.placeholder(tf.int32, [None, 100])
        self.q_tokens_len = tf.placeholder(tf.int32, [None])
        self.rel_tokens = tf.placeholder(tf.int32, [None, 20])
        self.rel_tokens_len = tf.placeholder(tf.int32, [None])
        self.rel_splitted_tokens = tf.placeholder(tf.int32, [None, 20])
        self.rel_splitted_tokens_len = tf.placeholder(tf.int32, [None])
        self.max_pooling_type = params['max_pooling']
        
        self.rel_tokens_neg = tf.placeholder(tf.int32, [None, 20])
        self.rel_tokens_len_neg = tf.placeholder(tf.int32, [None])
        self.rel_splitted_tokens_neg = tf.placeholder(tf.int32, [None, 20])
        self.rel_splitted_tokens_len_neg = tf.placeholder(tf.int32, [None])
        self.learning_rate = params['learning_rate']
        self.grad_clip_norm = params['grad_clip_norm']
        self.gamma = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.token_vocab_size = params['token_vocab_size']
        self.token_embedding_size = params['token_embedding_size']
        self.relation_vocab_size = params['relation_vocab_size']
        self.relation_embedding_size = params['relation_embedding_size']
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.sess = sess
        
        tf.global_variables_initializer()
        
        self._embedding()
        self._build_model()
        self._build_distance()
        
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        
        
    def _embedding(self):
        self.token_embedding_matrix = tf.get_variable("token_embedding",shape=[self.token_vocab_size,
                                            self.token_embedding_size],
                                            dtype="float",
                                            trainable=True)
        
        self.relation_embedding_matrix = tf.get_variable("relation_embedding",shape=[self.relation_vocab_size,
                                            self.relation_embedding_size],
                                            dtype="float",
                                            trainable=True)
        
        self.q_tokens_embeddings = tf.nn.embedding_lookup(self.token_embedding_matrix, self.q_tokens)
        self.rel_splitted_tokens_embeddings = tf.nn.embedding_lookup(self.token_embedding_matrix, self.rel_splitted_tokens)
        self.rel_splitted_tokens_neg_embeddings = tf.nn.embedding_lookup(self.token_embedding_matrix, self.rel_splitted_tokens_neg)
        
        self.rel_tokens_embeddings = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.rel_tokens)
        self.rel_tokens_neg_embeddings = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.rel_tokens_neg)
        
        
    def _build_shared_layer(self):
        
        with tf.variable_scope("bilstm") as scope:
        
            [question_embed, 
            question_states] = _build_bi_lstm(self.q_tokens_embeddings, self.q_tokens_len, self.hidden_size, scope="shared_layer")
            scope.reuse_variables()
                
            [relation_embed, 
            relation_states] = _build_bi_lstm(self.rel_tokens_embeddings, self.rel_tokens_len, self.hidden_size, scope="shared_layer")
            scope.reuse_variables()
                
            [relation_splited_embed, 
            relation_splitted_states] = _build_bi_lstm(self.rel_splitted_tokens_embeddings, 
                                                            self.rel_splitted_tokens_len, 
                                                            self.hidden_size, 
                                                            initial_state_fw=relation_states[0], 
                                                            initial_state_bw=relation_states[1],
                                                            scope="shared_layer")
            scope.reuse_variables()
            
            [relation_embed_neg, 
            relation_states_neg] = _build_bi_lstm(self.rel_tokens_neg_embeddings, 
                                                 self.rel_tokens_len_neg, self.hidden_size, scope="shared_layer")
            scope.reuse_variables()
                
            [relation_splited_embed_neg, 
            relation_splitted_states_neg] = _build_bi_lstm(self.rel_splitted_tokens_neg_embeddings, 
                                                            self.rel_splitted_tokens_len_neg, 
                                                            self.hidden_size, 
                                                            initial_state_fw=relation_states_neg[0], 
                                                            initial_state_bw=relation_states_neg[1],
                                                            scope="shared_layer")
            return [question_embed, question_states, 
                  relation_embed, relation_splited_embed, 
                  relation_embed_neg, relation_splited_embed_neg]
        
    def _build_model(self):

         with tf.variable_scope("hr_bilstm") as scope:
 
            [question_embed, question_states, 
            relation_embed, relation_splited_embed, 
            relation_embed_neg, relation_splited_embed_neg] = self._build_shared_layer()
        
            relations = tf.concat([relation_embed, relation_splited_embed], axis=1) # concat along the axis of time, batch x time x dims 
            relation_representation = tf.reduce_max(relations, axis=1)
        
            relations_neg = tf.concat([relation_embed_neg, relation_splited_embed_neg], axis=1) # concat along the axis of time, batch x time x dims 
            relation_representation_neg = tf.reduce_max(relations_neg, axis=1)

            question_embed_, _ = _build_bi_lstm(question_embed, self.q_tokens_len, self.hidden_size, 
                                         initial_state_fw=question_states[0], 
                                         initial_state_bw=question_states[1], 
                                         scope="second_layer")
        
            if self.max_pooling_type == "sum_max":
                question_representation = tf.reduce_max(question_embed + question_embed_, axis=1) # batch  x dims 

            elif self.max_pooling_type == "max_sum":            
                question_representation = tf.reduce_max(question_embed, axis=1) + tf.reduce_max(question_embed_, axis=1)
            
            self.relation_representation = relation_representation
            self.relation_representation_neg = relation_representation_neg
            self.question_representation = question_representation
        
    def _build_distance(self):
        positive_distance = tf.multiply(self.relation_representation, self.question_representation)
        positive_distance = tf.reduce_sum(positive_distance, axis=1, keep_dims=True) # sum over the data dimension not batch dimension
        positive_distance = tf.div(positive_distance, tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.relation_representation),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.question_representation),1,keep_dims=True))))
        
        self.positive_distance = tf.reshape(positive_distance, [-1], name="positive_distance")

        neg_distance = tf.multiply(self.relation_representation_neg, self.question_representation)
        neg_distance = tf.reduce_sum(neg_distance, axis=1, keep_dims=True) # sum over the data dimension not batch dimension
        neg_distance = tf.div(neg_distance, tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.relation_representation_neg),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.question_representation),1,keep_dims=True))))
        
        self.neg_distance = tf.reshape(neg_distance, [-1], name="negative_distance")
        
    def _save(self, model_dir, model_prefix):
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        
    def _restore(self, model_dir, model_prefix):
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
    
    def _build_loss(self):

        self.loss = tf.maximum(0, self.gamma - self.positive_distance + self.neg_distance)
        
        self.total_loss = -tf.reduce_mean(self.loss)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.total_loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        
    def _train(self, q_tokens, q_tokens_len, rel_tokens, rel_tokens_len, 
            rel_splitted_tokens, rel_splitted_tokens_len, rel_tokens_neg, rel_tokens_len_neg,
             rel_splitted_tokens_neg, rel_splitted_tokens_len_neg):
        feed_dict = {self.q_tokens:q_tokens, self.q_tokens_len:q_tokens_len, self.rel_tokens:rel_tokens,
                 self.rel_tokens_len:rel_tokens_len, self.rel_splitted_tokens:rel_splitted_tokens,
                 self.rel_splitted_tokens_len:rel_splitted_tokens_len, self.rel_tokens_neg:rel_tokens_neg,
                 self.rel_tokens_len_neg:rel_tokens_len_neg, self.rel_splitted_tokens_neg:rel_splitted_tokens_neg, 
                 self.rel_splitted_tokens_len_neg:rel_splitted_tokens_len_neg
        }
        
        _, step, loss = self.sess.run([self.train_op, 
                             self.global_step, 
                             self.total_loss],  
                             feed_dict)
        return loss
    
    
    def _infer(self, q_tokens, q_tokens_len, rel_tokens, rel_tokens_len, rel_splitted_tokens, rel_splitted_tokens_len):
        
        predicted_score = tf.reduce_mean(self.positive_distance)

        feed_dict = {self.q_tokens:q_tokens, self.q_tokens_len:q_tokens_len, 
                 self.rel_tokens:rel_tokens, self.rel_tokens_len:rel_tokens_len, 
                 self.rel_splitted_tokens:rel_splitted_tokens, self.rel_splitted_tokens_len:rel_splitted_tokens_len,
                 self.rel_tokens_neg:rel_tokens, self.rel_tokens_len_neg:rel_tokens_len, 
                 self.rel_splitted_tokens_neg:rel_splitted_tokens, self.rel_splitted_tokens_len:rel_splitted_tokens_len
        }
        
        predicted_score = self.sess.run([predicted_score],  
                             feed_dict)
        return predicted_score
            
            
            
        
        
        
        
    
