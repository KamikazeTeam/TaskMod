from __future__ import print_function
from __future__ import division

from keras.models import Model
from keras.layers import GlobalAveragePooling1D, Dense
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Activation, Dropout, merge, Merge, Flatten # merge for tensor Merge for layer
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2

from . import AbstractMod

class CNNMod(AbstractMod):
    def add_conf(self,c):
        # model-external:
        c['nb_epoch']   = 8#3#12
        c['patience']   = 8#3#12
        c['batch_size'] = 904#96#1024#64
        c['num_batchs'] = 12#113#9#10848=32*3*113
        c['showbatchs'] = False#10#False
        #c['debug']   = True
        #c['verbose'] = 1
        #c['nb_epoch']= 1#6
        # dict
        #c['embdict']       = 'W2V'
        #c['embname']       = 'GoogleNews-vectors-negative%d.txt'
        c['embdict']       = 'GloVe'#'W2V'
        c['embname']       = 'glove.6B'#'GoogleNews-vectors-negative%d.bin.gz'
        c['embdim']        = 300
        #c['embname']       = 'polyglot.es.%d.txt'
        #c['embdim']        = 64
        c['embprune']      = 0
        c['embicase']      = True#False
        c['inp_e_dropout'] = 0
        c['inp_w_dropout'] = 0
        c['maskzero']  = False
        c['spad']      = 30#60
        c['spad0']     = 30
        c['spad1']     = 30
        # mlp
        c['Ddim']    = list([1])#list([2,1])#list([])
        c['Dinit']   = 'he_uniform'#'glorot_uniform'#'he_normal''he_uniform'
        c['Dact']    = 'relu'
        c['Dl2reg']  = 1e-5
        # dense dim=N
        c['nndeep'] = 0
        c['nninit'] = 'he_uniform'
        c['nnact']  = 'relu'#'tanh'
        c['nnl2reg']= 1e-5
        # cnn:
        c['cdim']       = {1: 1}#, 2: 1/2}#, 3: 1/2, 4: 1/2}
        c['cinit']      = 'he_uniform'#'glorot_uniform'#'he_normal''he_uniform'
        c['cact']       = 'relu'
        c['cl2reg']     = 1e-5#4
        c['cdropout']   = 0

    def create_model(self, mergedinp0, mergedinp1, N, c):
        con=[]
        dpt=[]
        gmp=[]
        flt=[]
        for fl, cd in c['cdim'].items():
            nb_filter = int(N*cd)
            con.append(Convolution1D(name='con%d'%(fl,),
                                     nb_filter=nb_filter,
                                     filter_length=fl,
                                     border_mode='valid',#valid same
                                     activation=c['cact'],
                                     init=c['cinit'],
                                     W_regularizer=l2(c['cl2reg'])))
                                     #subsample_length=1))
            dpt.append(Dropout(name='dpt%d'%(fl,),p=c['cdropout']))
            gmp.append(MaxPooling1D(name='gmp%d'%(fl,),
                                    pool_length=int(c['spad']-fl+1)))#-fl+1
            flt.append(Flatten(name='flt%d'%(fl,)))
        convol0=[]
        convol1=[]
        for fl in range(len(c['cdim'])):
            convol0.append(con[fl](mergedinp0))
            convol1.append(con[fl](mergedinp1))
        drpout0=[]
        drpout1=[]
        for fl in range(len(c['cdim'])):
            drpout0.append(dpt[fl](convol0[fl]))
            drpout1.append(dpt[fl](convol1[fl]))
        gmpool0=[]
        gmpool1=[]
        for fl in range(len(c['cdim'])):
            gmpool0.append(gmp[fl](drpout0[fl]))
            gmpool1.append(gmp[fl](drpout1[fl]))
        flated0=[]
        flated1=[]
        for fl in range(len(c['cdim'])):
            flated0.append(flt[fl](gmpool0[fl]))
            flated1.append(flt[fl](gmpool1[fl]))
        if len(c['cdim']) > 1:
            merged0=merge(flated0,name='merged0',mode='concat')
            merged1=merge(flated1,name='merged1',mode='concat')
        else:
            merged0=flated0[0]
            merged1=flated1[0]
        dense = []
        for i in range(c['nndeep']):
            dense.append(Dense(name='nndeep%d'%(i,), output_dim=N, activation=c['nnact'], init=c['nninit'], W_regularizer=l2(c['nnl2reg'])))
        for i in range(c['nndeep']):
            merged0 = dense[i](merged0)
            merged1 = dense[i](merged1)
        self.testlayers.append(convol0[0])
        self.testlayers.append(drpout0[0])
        self.testlayers.append(gmpool0[0])
        self.testlayers.append(flated0[0])
        self.testlayers.append(merged0)
        return merged0, merged1
        #Activation('linear') Dropout

def mod():
    return CNNMod()
