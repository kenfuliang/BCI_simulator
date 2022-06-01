
import numpy as np
from keras.layers import Input,SimpleRNN,Lambda,Dense, LSTM#, Dense, Flatten, Permute, ,
import keras
from datetime import date
from keras.backend import exp
from keras.models import Model
from keras.layers import SimpleRNNCell
import keras.backend as K
import tensorflow as tf
import os

def get_loss(mask_value=999):
    mask_value = K.variable(mask_value)
    def masked_poisson_loss(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character '#'
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        # multiply categorical_crossentropy with the mask
        loss = keras.losses.poisson(y_true, y_pred) * mask
        #keras.losses.poisson
        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)
    return masked_poisson_loss


def TrainStatefulSeq2SeqRNN_2(data,date
            ,path='../saved/encoder/JenkinsC/'
            ,model_arch=[192,192,192]
            ,l2_ke = 0,l2_re = 0
            ,lr=0.01,decay=0.01
            ,M1_delay=0,PMd_delay=0
            ,ep=2,epochs=1000
            ,verbose=False,post_fix=''):
    '''
    Training data are {(centerOut, centerIn), (centerOut,centerIn), ...}
    '''
    from keras.layers import Masking 
    from keras.preprocessing.sequence import pad_sequences

    assert M1_delay!=0
    assert PMd_delay!=0

    df = data.df
    dt = data.dt
    numBins = data.numBins
    num_output = 192

    rt_hand_p = []
    rt_hand_v = []
    rt_cursor_p = []
    rt_cursor_v = []
    rt_n = []
    for trialIdx in data.centerOutTrials:
        centerOutIdx = trialIdx
        centerInIdx = trialIdx+1
        if centerInIdx not in df.index:
            next
        else:
            hand_p = np.vstack([df['binHandPos'][centerOutIdx][:,0:2],df['binHandPos'][centerInIdx][:,0:2]])
            hand_v = np.vstack([df['binHandVel'][centerOutIdx][:,0:2],df['binHandVel'][centerInIdx][:,0:2]])

            M1 = np.vstack([df['binSpike'][centerOutIdx-1][-M1_delay:,:96],
                            df['binSpike'][centerOutIdx][:,:96],
                            df['binSpike'][centerInIdx][:-M1_delay,:96]]
                          )

            PMd = np.vstack([df['binSpike'][centerOutIdx-1][-PMd_delay:,96:],
                            df['binSpike'][centerOutIdx][:,96:],
                            df['binSpike'][centerInIdx][:-PMd_delay,96:]]
                          )
            n = np.hstack([M1,PMd])

            rt_hand_p.append(hand_p)
            rt_hand_v.append(hand_v)
            rt_n.append(n)
        
    rt_hand_p = pad_sequences(rt_hand_p,padding='post',value=9999)
    rt_hand_v = pad_sequences(rt_hand_v,padding='post',value=9999)
    rt_n = pad_sequences(rt_n,padding='post',value=9999)

    x_train = np.concatenate([rt_hand_p,rt_hand_v],axis=2)
    y_train = rt_n

    assert x_train.shape[:2]==y_train.shape[:2],"{} and {}".format(x_train.shape, y_train.shape)

    #x_train = x_train[:,:50]
    #y_train = y_train[:,:50]
    

    batch_size=x_train.shape[0]

    inpt = Input(batch_shape=x_train.shape)
    
    x = Masking(mask_value=(9999,9999,9999,9999))(inpt)
    x = SimpleRNN(model_arch[0], activation='sigmoid'
                      ,kernel_regularizer=keras.regularizers.l2(l2_ke)
                      ,recurrent_regularizer=keras.regularizers.l2(l2_re)
                      ,stateful=False
                      ,return_sequences=True)(x)
    x = Dense(model_arch[1], activation='sigmoid')(x)
    x = Dense(model_arch[2], activation='sigmoid')(x)
    x = Dense(num_output)(x)
    x = Lambda(lambda x: exp(x))(x)
    
    model = Model(inpt,x)
    model.summary()
    model.compile(loss=get_loss(mask_value=9999)#keras.losses.poisson
                  ,optimizer=keras.optimizers.Adam(lr=lr,decay=decay)
                 )

    
    for ii in range(1,ep):
        #print("training epoch ",ii)
        for _ in range(epochs):
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      verbose=verbose,
                      shuffle=False,
                      epochs=1)
            model.reset_states()

        inpt = Input(batch_shape=[1,1,4])

        x = Masking(mask_value=(9999,9999,9999,9999))(inpt)
        x = SimpleRNN(model_arch[0], activation='sigmoid'
                          ,kernel_regularizer=keras.regularizers.l2(l2_ke)
                          ,recurrent_regularizer=keras.regularizers.l2(l2_re)
                          ,stateful=True
                          ,return_sequences=False)(x)
        x = Dense(model_arch[1], activation='sigmoid')(x)
        x = Dense(model_arch[2], activation='sigmoid')(x)
        x = Dense(num_output)(x)
        x = Lambda(lambda x: exp(x))(x)
        rt_model = Model(inpt,x)

        for nb, layer in enumerate(model.layers):
            rt_model.layers[nb].set_weights(layer.get_weights())

        model_path = os.path.join(path,"Seq2Seq_stateful2_b{}_{}_r{}_decay{}_ke{}_re{}_delayed{}{}_extraBins{}_{}_ep{}.h5".format(dt,date,model_arch[0],decay,l2_ke,l2_re,M1_delay,PMd_delay,data.extraBins,post_fix,ii))
        #model_path = os.path.join(path,"test.h5")
        rt_model.save(model_path)
        print("Save {}".format(model_path))

def TrainStatefulSeq2SeqLSTM(data,date
            ,path='../saved/encoder/JenkinsC/'
            ,model_arch=[192,192,192]
            ,l2_ke = 0,l2_re = 0
            ,lr=0.01,decay=0.01
            ,M1_delay=0,PMd_delay=0
            ,ep=2,epochs=1000
            ,verbose=False,post_fix=''):

    assert M1_delay!=0
    assert PMd_delay!=0

    df = data.df
    dt = data.dt
    numBins = data.numBins
    num_output = 192

    hand_p = np.concatenate(df['binHandPos'],axis=0)
    hand_v = np.concatenate(df['binHandVel'],axis=0)
    x_train = np.concatenate([hand_p,hand_v],axis=1)
    n = np.concatenate(df['binSpike'],axis=0)

    max_delay = max(M1_delay, PMd_delay)
    x_train = x_train[max_delay:]
    y_train = np.concatenate( [n[max_delay-M1_delay:-M1_delay,:96], n[max_delay-PMd_delay:-PMd_delay,96:]], axis=1)
    
    x_train = np.expand_dims(x_train,axis=0) 
    y_train = np.expand_dims(y_train,axis=0) 

    assert x_train.shape[:2]==y_train.shape[:2],"{} and {}".format(x_train.shape, y_train.shape)

    inpt = Input(batch_shape=x_train.shape)
    
    x = LSTM(model_arch[0], activation='sigmoid'
                      ,kernel_regularizer=keras.regularizers.l2(l2_ke)
                      ,recurrent_regularizer=keras.regularizers.l2(l2_re)
                      ,stateful=False
                      ,return_sequences=True)(inpt)
    x = Dense(model_arch[1], activation='sigmoid')(x)
    x = Dense(model_arch[2], activation='sigmoid')(x)
    x = Dense(num_output)(x)
    x = Lambda(lambda x: exp(x))(x)
    
    model = Model(inpt,x)
    model.compile(loss=keras.losses.poisson
                  ,optimizer=keras.optimizers.Adam(lr=lr,decay=decay)
                 )

    
    for ii in range(1,ep):
        print("training epoch ",ii)
        model.fit(x_train, y_train,
                      batch_size=1,
                      verbose=verbose,
                      shuffle=False,
                      epochs=epochs)

        inpt = Input(batch_shape=[1,1,4])
        x = LSTM(model_arch[0], activation='sigmoid'
                          , kernel_regularizer=keras.regularizers.l2(l2_ke)
                          , recurrent_regularizer=keras.regularizers.l2(l2_re)
                          ,stateful=True
                          ,return_sequences=False)(inpt)
        x = Dense(model_arch[1], activation='sigmoid')(x)
        x = Dense(model_arch[2], activation='sigmoid')(x)
        x = Dense(num_output)(x)
        x = Lambda(lambda x: exp(x))(x)
        rt_model = Model(inpt,x)

        for nb, layer in enumerate(model.layers):
            rt_model.layers[nb].set_weights(layer.get_weights())

        model_path = path+"Seq2Seq_LSTM_b{}_{}_r{}_decay{}_ke{}_re{}_delayed{}{}_extraBins{}_{}_ep{}.h5".format(dt,date,model_arch[0],decay,l2_ke,l2_re,M1_delay,PMd_delay,data.extraBins,post_fix,ii)
        rt_model.save(model_path)
        print("Save {}".format(model_path))




def TrainStatefulSeq2SeqRNN(data,date
            ,path='../saved/encoder/JenkinsC/'
            ,model_arch=[192,192,192]
            ,l2_ke = 0,l2_re = 0
            ,lr=0.01,decay=0.01
            ,M1_delay=0,PMd_delay=0
            ,ep=2,epochs=1000
            ,verbose=False,post_fix=''):


    df = data.df
    dt = data.dt
    numBins = data.numBins
    num_output = 192

    hand_p = np.concatenate(df['binHandPos'],axis=0)
    hand_v = np.concatenate(df['binHandVel'],axis=0)
    x_train = np.concatenate([hand_p,hand_v],axis=1)
    n = np.concatenate(df['binSpike'],axis=0)

    max_delay = max(M1_delay, PMd_delay)
    x_train = x_train[max_delay:]
    y_train = np.concatenate( [n[max_delay-M1_delay:len(n)-M1_delay,:96], n[max_delay-PMd_delay:len(n)-PMd_delay,96:]], axis=1)
    
    x_train = np.expand_dims(x_train,axis=0) 
    y_train = np.expand_dims(y_train,axis=0) 

    assert x_train.shape[:2]==y_train.shape[:2],"{} and {}".format(x_train.shape, y_train.shape)

    inpt = Input(batch_shape=x_train.shape)
    
    x = SimpleRNN(model_arch[0], activation='sigmoid'
                      ,kernel_regularizer=keras.regularizers.l2(l2_ke)
                      ,recurrent_regularizer=keras.regularizers.l2(l2_re)
                      ,stateful=True
                      ,return_sequences=True)(inpt)
    x = Dense(model_arch[1], activation='sigmoid')(x)
    x = Dense(model_arch[2], activation='sigmoid')(x)
    x = Dense(num_output)(x)
    x = Lambda(lambda x: exp(x))(x)
    
    model = Model(inpt,x)
    model.compile(loss=keras.losses.poisson
                  ,optimizer=keras.optimizers.Adam(lr=lr,decay=decay)
                 )

    
    for ii in range(1,ep):
        print("training epoch ",ii)
        for _ in range(epochs):
            model.fit(x_train, y_train,
                      batch_size=1,
                      verbose=verbose,
                      shuffle=False,
                      epochs=1)
            model.reset_states()

        inpt = Input(batch_shape=[1,1,4])
        x = SimpleRNN(model_arch[0], activation='sigmoid'
                          , kernel_regularizer=keras.regularizers.l2(l2_ke)
                          , recurrent_regularizer=keras.regularizers.l2(l2_re)
                          ,stateful=True
                          ,return_sequences=False)(inpt)
        x = Dense(model_arch[1], activation='sigmoid')(x)
        x = Dense(model_arch[2], activation='sigmoid')(x)
        x = Dense(num_output)(x)
        x = Lambda(lambda x: exp(x))(x)
        rt_model = Model(inpt,x)

        for nb, layer in enumerate(model.layers):
            rt_model.layers[nb].set_weights(layer.get_weights())

        model_path = path+"Seq2Seq_stateful_b{}_{}_r{}_decay{}_ke{}_re{}_delayed{}{}_extraBins{}_{}_ep{}.h5".format(dt,date,model_arch[0],decay,l2_ke,l2_re,M1_delay,PMd_delay,data.extraBins,post_fix,ii)
        rt_model.save(model_path)
        print("Save {}".format(model_path))


def TrainRNN(data,date
            ,path='../saved/encoder/JenkinsC/'
            ,model_arch=[192,192,192]
            ,l2_ke = 0,l2_re = 0,decay=0.01,M1_delay=0,PMd_delay=0
            ,ep=2,batch_size=128,epochs=1000
            ,verbose=False,post_fix=''):

    dt = data.dt
    train_data = data.next_train_batch(batch_size=500,replace=False,M1_delay=M1_delay,PMd_delay=PMd_delay)
    x_train = np.concatenate([train_data['hand_p'],train_data['hand_v']],axis=2)
    y_train = train_data['n']
    
    print('Input shape:',x_train.shape,', Output shape"',y_train.shape)

    num_input = 4
    num_output= data.numChannels

    input_shape = [data.extraBins+1, num_input]
    inpt = Input(shape=input_shape)


    x = SimpleRNN(model_arch[0], activation='sigmoid'
                      , kernel_regularizer=keras.regularizers.l2(l2_ke)
                      , recurrent_regularizer=keras.regularizers.l2(l2_re))(inpt)
    x = Dense(model_arch[1], activation='sigmoid')(x)
    x = Dense(model_arch[2], activation='sigmoid')(x)
    x = Dense(num_output)(x)
    x = Lambda(lambda x: exp(x))(x)

    model = Model(inpt,x)
    model.summary()
    model.compile(loss=keras.losses.poisson
                  ,optimizer=keras.optimizers.Adam(lr=0.01,decay=decay)
                 )

    for ii in range(1,ep):
        print("training epoch ",ii)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  verbose=verbose,
                  epochs=epochs)

        saved_model = path+"RNN_b{}_{}_r{}_decay{}_ke{}_bs{}_delayed{}{}_extraBins{}_{}_ep{}.h5".format(dt,date,model_arch[0],decay,l2_ke,batch_size,M1_delay,PMd_delay,data.extraBins,post_fix,ii)
        model.save(saved_model)
        print("Save {}".format(saved_model))

def TrainNoiseRNN(data,path='../saved/encoder/JenkinsC/',date=date.today().strftime("%Y%m%d"),stddev=0,l2_ke = 0,l2_re = 0,decay=0.01,ep=2,batch_size=128,epochs=1000,M1_delay=0,PMd_delay=0,verbose=False,model_arch=[192,192,192],post_fix=''):
    dt = data.dt
    train_data = data.next_train_batch(batch_size=500,replace=False,M1_delay=M1_delay,PMd_delay=PMd_delay)
    x_train = np.concatenate([train_data['hand_p'],train_data['hand_v']],axis=2)
    y_train = train_data['n']

    #x_train = np.concatenate([train_data['hand_p'],train_data['hand_v']],axis=2)
    #y_train = train_data['n'][:,-1]
    #
    #max_delay = max(M1_delay,PMd_delay)
    #print("max_delay:",max_delay)
    #x_train = x_train[max_delay:]
    #y_train = np.concatenate([y_train[max_delay-M1_delay:len(y_train)-M1_delay,:96],y_train[max_delay-PMd_delay:len(y_train)-PMd_delay,96:]],axis=1)
    #
    print('Input shape:',x_train.shape,', Output shape"',y_train.shape)

    num_input = 4
    num_output= data.numChannels

    input_shape = [data.extraBins+1, num_input]
    inpt = Input(shape=input_shape)


    #x = SimpleRNN(model_arch[0], activation='sigmoid'
    #                  , kernel_regularizer=keras.regularizers.l2(l2_ke)
    #                  , recurrent_regularizer=keras.regularizers.l2(l2_re))(inpt)
    x = keras.layers.RNN(SimpleNoiseRNNCell(units=model_arch[0],stddev=stddev
                      , activation='sigmoid'
                      , kernel_regularizer=keras.regularizers.l2(l2_ke)
                      , recurrent_regularizer=keras.regularizers.l2(l2_re)))(inpt)

    #x = keras.layers.RNN(SimpleNoiseRNNCell(units=model_arch[0],stddev=stddev))(inpt)




    x = Dense(model_arch[1], activation='sigmoid')(x)
    x = Dense(model_arch[2], activation='sigmoid')(x)
    x = Dense(num_output)(x)
    x = Lambda(lambda x: exp(x))(x)

    model = Model(inpt,x)
    model.summary()
    model.compile(loss=keras.losses.poisson
                  ,optimizer=keras.optimizers.Adam(lr=0.01,decay=decay)
                 )

    for ii in range(1,ep):
        print("training epoch ",ii)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  verbose=verbose,
                  epochs=epochs)

        saved_model = path+"RNN_b{}_{}_r{}_decay{}_ke{}_re{}_bs{}_delayed{}{}_extraBins{}_noise{}_{}_ep{}.h5".format(dt,date,model_arch[0],decay,l2_ke,l2_re,batch_size,M1_delay,PMd_delay,data.extraBins,stddev,post_fix,ii)
        model.save(saved_model)
        print("Save {}".format(saved_model))


class SimpleNoiseRNNCell(SimpleRNNCell):
    def __init__(self,stddev=0,**kwargs):
        super(SimpleNoiseRNNCell, self).__init__(**kwargs)
        self.stddev=stddev

    #def call(self, inputs, states, training=None):
    #    prev_output = states[0] if tf.nest.is_nested(states) else states
    #    h = backend.dot(inputs, self.kernel)
    #    output = h + backend.dot(prev_output, self.recurrent_kernel)
    #    
    #    ## add Gaussian noise here
    #    print(output.shape)
    #    #output = tf.random.normal(shape=[192],mean=output, stddev=self.stddev)
    #    
    #    if self.activation is not None:
    #        output = self.activation(output)
    #    
    #    new_state = [output] if tf.nest.is_nested(states) else output
    #    return output, new_state

    def call(self, inputs, states, training=None):
        prev_output = states[0]
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(prev_output),
                self.recurrent_dropout,
                training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output *= rec_dp_mask
        output = h + K.dot(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = self.activation(output)

        ## add Gaussian noise here
        output = tf.random.normal(shape=[192],mean=output, stddev=self.stddev)
 

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                output._uses_learning_phase = True
        return output, [output]



#    def call(self, inputs, states, training=None):
#        print(self)
#        prev_output = states[0] if tf.nest.is_nested(states) else states
#        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
#        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
#            prev_output, training)
#        
#        if dp_mask is not None:
#          h = backend.dot(inputs * dp_mask, self.kernel)
#        else:
#          h = backend.dot(inputs, self.kernel)
#        if self.bias is not None:
#          h = backend.bias_add(h, self.bias)
#        
#        if rec_dp_mask is not None:
#          prev_output = prev_output * rec_dp_mask
#        output = h + backend.dot(prev_output, self.recurrent_kernel)
#        
#        ## add Gaussian noise here
#        output = tf.random.normal(mean=output, stddev=self.stddev)
#        
#        
#        if self.activation is not None:
#          output = self.activation(output)
#        
#        new_state = [output] if tf.nest.is_nested(states) else output
#        return output, new_state


#class SimpleNoiseRNNCell(DropoutRNNCellMixin, base_layer.BaseRandomLayer):
#    """Inherited from SimpleRNNCell
#    """
#    def __init__(self,
#                 units,
#                 stddev=0,
#                 activation='tanh',
#                 use_bias=True,
#                 kernel_initializer='glorot_uniform',
#                 recurrent_initializer='orthogonal',
#                 bias_initializer='zeros',
#                 kernel_regularizer=None,
#                 recurrent_regularizer=None,
#                 bias_regularizer=None,
#                 kernel_constraint=None,
#                 recurrent_constraint=None,
#                 bias_constraint=None,
#                 dropout=0.,
#                 recurrent_dropout=0.,
#                 **kwargs):
#        #if units < 0:
#        #    raise ValueError(f'Received an invalid value for argument `units`, '
#        #                   f'expected a positive integer, got {units}.')
#        # By default use cached variable under v2 mode, see b/143699808.
#        #if tf.compat.v1.executing_eagerly_outside_functions():
#        #    self._enable_caching_device = kwargs.pop('enable_caching_device', True)
#        #else:
#        #    self._enable_caching_device = kwargs.pop('enable_caching_device', False)
#        super(SimpleRNNCell, self).__init__(**kwargs)
#        self.stddev = stddev
#        self.units = units
#        self.activation = activations.get(activation)
#        self.use_bias = use_bias
#        
#        self.kernel_initializer = initializers.get(kernel_initializer)
#        self.recurrent_initializer = initializers.get(recurrent_initializer)
#        self.bias_initializer = initializers.get(bias_initializer)
#        
#        self.kernel_regularizer = regularizers.get(kernel_regularizer)
#        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
#        self.bias_regularizer = regularizers.get(bias_regularizer)
#        
#        self.kernel_constraint = constraints.get(kernel_constraint)
#        self.recurrent_constraint = constraints.get(recurrent_constraint)
#        self.bias_constraint = constraints.get(bias_constraint)
#        
#        self.dropout = min(1., max(0., dropout))
#        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
#        self.state_size = self.units
#        self.output_size = self.units
#        
#        self.stddev = stddev
#        
#    #@tf_utils.shape_type_conversion
#    def build(self, input_shape):
#        default_caching_device = _caching_device(self)
#        self.kernel = self.add_weight(
#            shape=(input_shape[-1], self.units),
#            name='kernel',
#            initializer=self.kernel_initializer,
#            regularizer=self.kernel_regularizer,
#            constraint=self.kernel_constraint,
#            caching_device=default_caching_device)
#        self.recurrent_kernel = self.add_weight(
#            shape=(self.units, self.units),
#            name='recurrent_kernel',
#            initializer=self.recurrent_initializer,
#            regularizer=self.recurrent_regularizer,
#            constraint=self.recurrent_constraint,
#            caching_device=default_caching_device)
#        if self.use_bias:
#          self.bias = self.add_weight(
#              shape=(self.units,),
#              name='bias',
#              initializer=self.bias_initializer,
#              regularizer=self.bias_regularizer,
#              constraint=self.bias_constraint,
#              caching_device=default_caching_device)
#        else:
#          self.bias = None
#        self.built = True
#    
#    def call(self, inputs, states, training=None):
#        prev_output = states[0] if tf.nest.is_nested(states) else states
#        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
#        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
#            prev_output, training)
#        
#        if dp_mask is not None:
#          h = backend.dot(inputs * dp_mask, self.kernel)
#        else:
#          h = backend.dot(inputs, self.kernel)
#        if self.bias is not None:
#          h = backend.bias_add(h, self.bias)
#        
#        if rec_dp_mask is not None:
#          prev_output = prev_output * rec_dp_mask
#        output = h + backend.dot(prev_output, self.recurrent_kernel)
#        
#        ## add Gaussian noise here
#        output = tf.random.normal(mean=output, stddev=self.stddev)
#        
#        
#        if self.activation is not None:
#          output = self.activation(output)
#        
#        new_state = [output] if tf.nest.is_nested(states) else output
#        return output, new_state
#    
#    def get_config(self):
#        config = {
#            'units':
#                self.units,
#            'activation':
#                activations.serialize(self.activation),
#            'use_bias':
#                self.use_bias,
#            'kernel_initializer':
#                initializers.serialize(self.kernel_initializer),
#            'recurrent_initializer':
#                initializers.serialize(self.recurrent_initializer),
#            'bias_initializer':
#                initializers.serialize(self.bias_initializer),
#            'kernel_regularizer':
#                regularizers.serialize(self.kernel_regularizer),
#            'recurrent_regularizer':
#                regularizers.serialize(self.recurrent_regularizer),
#            'bias_regularizer':
#                regularizers.serialize(self.bias_regularizer),
#            'kernel_constraint':
#                constraints.serialize(self.kernel_constraint),
#            'recurrent_constraint':
#                constraints.serialize(self.recurrent_constraint),
#            'bias_constraint':
#                constraints.serialize(self.bias_constraint),
#            'dropout':
#                self.dropout,
#            'recurrent_dropout':
#                self.recurrent_dropout
#        }
#        config.update(_config_for_enable_caching_device(self))
#        base_config = super(SimpleRNNCell, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))

#class MinimalRNNCell(keras.layers.Layer):
#
#    def __init__(self, units, **kwargs):
#        self.units = units
#        self.state_size = units
#        super(MinimalRNNCell, self).__init__(**kwargs)
#
#    def build(self, input_shape):
#        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
#                                      initializer='uniform',
#                                      name='kernel')
#        self.recurrent_kernel = self.add_weight(
#                        shape=(self.units, self.units),
#                        initializer='uniform',
#                        name='recurrent_kernel')
#        self.built = True
#
#    def call(self, inputs, states):
#        prev_output = states[0]
#        h = K.dot(inputs, self.kernel)
#        output = h + K.dot(prev_output, self.recurrent_kernel)
#        return output, [output]
