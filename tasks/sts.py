from __future__ import print_function
from __future__ import division

import numpy as np
import random
import copy
#import os
import pickle
from nltk.tokenize import word_tokenize
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse
from collections import namedtuple

#from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Embedding, merge
from keras.regularizers import l2
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import backend as K

import pysts.embedding
import pysts.loader as loader
import pysts.vocab
import pysts.nlp

from . import AbstractTask

class STSTask(AbstractTask):
    def task_config(self):
        self.c['nb_runs'] = 1
        self.c['nb_epoch']   = 16#64
        self.c['patience']   = 4#16
        self.c['batch_size'] = 64#128
        self.c['num_batchs'] = 50#170
        self.c['showbatchs'] = False
        self.c['debug']      = False
        self.c['verbose']    = 2
        #self.c['epoch_fract']= 1/10
        self.c['embdict']       = 'GloVe'
        self.c['embdim']        = 300
        self.c['embprune']      = 100
        self.c['embicase']      = True#False
        self.c['inp_e_dropout'] = 0
        self.c['inp_w_dropout'] = 0
        self.c['maskzero']      = False
        #self.c['e_add_flags']   = True
        self.c['spad']      = 60
        #self.c['ptscorer']= None#B.mlp_ptscorer
        #self.c['mlpsum']  = 'absdiff'
        self.c['Ddim']    = 1#1#list([2,1])#2
        self.c['Dinit']   = 'he_uniform'#'glorot_uniform'#'he_normal''he_uniform'
        self.c['Dact']    = 'tanh'
        self.c['Dl2reg']  = 1e-5
        #self.c['f_add_kw'] = False
        self.c['fix_layers']= []  # mainly useful for transfer learning, or 'emb' to fix embeddings
        self.c['target']    = 'classes'#'classes''score'
        def pearsonobj(ny_true,ny_pred):
            my_true = K.mean(ny_true)
            my_pred = K.mean(ny_pred)
            var_true = (ny_true - my_true)**2
            var_pred = (ny_pred - my_pred)**2
            return - K.sum((ny_true - my_true) * (ny_pred - my_pred), axis=-1) / \
                     (K.sqrt(K.sum(var_true, axis=-1) * K.sum(var_pred, axis=-1)))    
        def Cpearsonobj(y_true, y_pred):
            ny_true = y_true[:,1] + 2*y_true[:,2] + 3*y_true[:,3] + 4*y_true[:,4] + 5*y_true[:,5]
            ny_pred = y_pred[:,1] + 2*y_pred[:,2] + 3*y_pred[:,3] + 4*y_pred[:,4] + 5*y_pred[:,5]
            return pearsonobj(ny_true,ny_pred)
        def Spearsonobj(y_true, y_pred):
            ny_true = y_true[:,0]
            ny_pred = y_pred[:,0]
            return pearsonobj(ny_true,ny_pred)
        if self.c['target'] == 'classes':
            self.c['loss']  = Cpearsonobj#'categorical_crossentropy'#O.pearsonobj
        if self.c['target'] == 'score':
            self.c['loss']  = Spearsonobj#'mse'#B.pearsonobj
        self.c['opt']   = 'adam'#'sgd'#'adadelta'#'adam'
        #self.c['balance_class'] = False
        self.c['gpu'] = 1.0
    def task_initial(self):
        self.name = 'sts'
        self.spad = self.c['spad']
        self.s0pad= self.spad
        self.s1pad= self.spad
        self.embed= None
        self.vocab= None
        #if self.c['gpu'] != 1.0:
        #    import tensorflow as tf
        #    def get_session(gpu_fraction):
        #        print('Setting gpu_fraction to %f' %gpu_fraction)
        #        num_threads = os.environ.get('OMP_NUM_THREADS')
        #        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        #        if num_threads: return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        #        else: return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #    K.tensorflow_backend._set_session(get_session(self.c['gpu']))
    def load_sts(self, filename, skip_unlabeled=True):
        s0 = []
        s1 = []
        labels = []
        lines=open(filename,'r').read().splitlines()
        for line in lines:
            line = line.rstrip()
            label, s0x, s1x = line.split('\t')
            if label == '':
                if skip_unlabeled: continue
                else: labels.append(-1.)
            else: labels.append(float(label))
            s0.append(word_tokenize(s0x))
            s1.append(word_tokenize(s1x))
        return (s0, s1, np.array(labels))
    def load_embed(self, embdict):
        if embdict=='GloVe':
            print('Load GloVe')
            self.embed = pysts.embedding.GloVe(N=self.c['embdim'])
    def load_vocab(self, vocabf):
        s0, s1, y  = self.load_sts(vocabf, skip_unlabeled=True) #need flexible
        self.vocab = pysts.vocab.Vocabulary(s0 + s1, prune_N=self.c['embprune'], icase=self.c['embicase'])
    def task_load_resources(self):
        if 'embdict' in self.c: self.load_embed(self.c['embdict'])
        if 'vocabf' in self.c : self.load_vocab(self.c['vocabf'])
    def task_load_data(self, fname):
        if fname is None: return None
        if self.vocab is None: self.load_vocab(fname)
        s0, s1, y= self.load_sts(fname, skip_unlabeled=True)
        y1  = np.array([s/5.0 for s in y])
        si0, sj0 = self.vocab.vectorize(s0, self.embed, spad=self.s0pad)
        si1, sj1 = self.vocab.vectorize(s1, self.embed, spad=self.s1pad)
        f0,   f1 = pysts.nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)
        gr = {'si0': si0, 'si1': si1, 'sj0': sj0, 'sj1': sj1, 'f0': f0, 'f1': f1,
              'classes': loader.sts_labels2categorical(y), 'score1': y1, 'score': y, 's0': s0, 's1': s1}
        return gr

    def task_build_model(self, mod_prep_model):
        K.clear_session()
        self.inputsint0 = Input(name='si0', shape=(self.s0pad,))
        self.inputsemb0 = Input(name='se0', shape=(self.s0pad, self.embed.N))
        self.inputsnlp0 = Input(name='f0', shape=(self.s0pad, pysts.nlp.flagsdim))
        self.inputsint1 = Input(name='si1', shape=(self.s1pad,))
        self.inputsemb1 = Input(name='se1', shape=(self.s1pad, self.embed.N))
        self.inputsnlp1 = Input(name='f1', shape=(self.s1pad, pysts.nlp.flagsdim))
        embmat = self.vocab.embmatrix(self.embed)
        embedi = Embedding(input_dim=embmat.shape[0], input_length=self.spad,
                           output_dim=self.embed.N, mask_zero=self.c['maskzero'],#True#False
                           weights=[embmat], trainable=True,
                           dropout=self.c['inp_w_dropout'], name='embedi')
        embediint0=embedi(self.inputsint0)
        embediint1=embedi(self.inputsint1)
        mergedemb0=merge([embediint0, self.inputsemb0], mode='sum', name='mergedemb0')
        mergedemb1=merge([embediint1, self.inputsemb1], mode='sum', name='mergedemb1')
        self.mergedinp0=merge([mergedemb0, self.inputsnlp0], mode='concat', name='mergedinp0')
        self.mergedinp1=merge([mergedemb1, self.inputsnlp1], mode='concat', name='mergedinp1')
        self.N = self.embed.N + pysts.nlp.flagsdim
        """if self.c['e_add_flags']:
            self.mergedinp0=merge([mergedemb0, self.inputsnlp0], mode='concat', name='mergedinp0')
            self.mergedinp1=merge([mergedemb1, self.inputsnlp1], mode='concat', name='mergedinp1')
            self.N = self.embed.N + pysts.nlp.flagsdim
        else:
            self.mergedinp0=mergedemb0
            self.mergedinp1=mergedemb1
            self.N = self.embed.N"""

        self.output0, self.output1 = mod_prep_model(self.mergedinp0, self.mergedinp1, self.N, self.s0pad, self.s1pad, self.c)
        absdiff = merge([self.output0,self.output1], mode=lambda X:K.abs(X[0] - X[1]), output_shape=lambda X: X[0], name='absdiff')
        muldiff = merge([self.output0,self.output1], mode='mul', name='muldiff')
        alldiff = merge([absdiff,muldiff], mode='concat', name='alldiff')
        mlpdiff = alldiff
        Ddim  =self.c['Ddim']
        if Ddim == 0:
            Ddim = []
        elif not isinstance(Ddim, list):
            Ddim = [Ddim]
        if Ddim:
            for i, D in enumerate(Ddim):
                dendiff = Dense(name='mlp'+'hdn%d'%(i,), 
                                output_dim=int(self.N*D), 
                                W_regularizer=l2(self.c['Dl2reg']), 
                                activation=self.c['Dact'], 
                                init=self.c['Dinit'])(mlpdiff)
                mlpdiff = dendiff
        self.classes = Dense(name='classes', 
                             output_dim=6, 
                             W_regularizer=l2(self.c['Dl2reg']), 
                             activation='softmax', 
                             init=self.c['Dinit'])(mlpdiff)
        model = Model(input=[self.inputsint0,self.inputsint1,self.inputsemb0,self.inputsemb1,self.inputsnlp0,self.inputsnlp1],
                      output=self.classes)

        for lname in self.c['fix_layers']:
            model.nodes[lname].trainable = False
        if self.c['target'] == 'classes':
            model.compile(loss={'classes': self.c['loss']}, optimizer=self.c['opt'])
        if self.c['target'] == 'score':
            model.compile(loss={'score1': self.c['loss']}, optimizer=self.c['opt'])
        print(model.layers)
        print("Constructed!")
        self.model=model
        return model
    def task_debug_model(self, mod_debug_model):
        mod_debug_model(self.mergedinp0, self.mergedinp1, self.N, self.s0pad, self.s1pad, self.c, self.output0, self.output1,
                        self.inputsint0,self.inputsint1,self.inputsemb0,self.inputsemb1,self.inputsnlp0,self.inputsnlp1, self.classes,
                        self.sample_pairs, self.grt)#use self.grt

    def sample_pairs(self, gr, batch_size, shuffle=True, once=False):
        num = len(gr['score'])
        idN = int((num+batch_size-1) / batch_size)
        ids = list(range(num))
        while True:
            if shuffle:
                random.shuffle(ids)
            grr= copy.deepcopy(gr)
            for name, value in grr.items():
                valuer=copy.copy(value)
                for i in range(num):
                    valuer[i]=value[ids[i]]
                grr[name] = valuer
            for i in range(idN):
                sl  = slice(i*batch_size, (i+1)*batch_size)
                grsl= dict()
                for name, value in grr.items():
                    grsl[name] = value[sl]
                grsl['se0'] = self.embed.map_jset(grsl['sj0'])
                grsl['se1'] = self.embed.map_jset(grsl['sj1'])
                x = [grsl['si0'],grsl['si1'],grsl['se0'],grsl['se1'],grsl['f0'],grsl['f1']]
                if self.c['target'] == 'classes':
                    y = grsl['classes']
                if self.c['target'] == 'score':
                    y = grsl['score1']
                yield (x,y)
                #yield grsl
            if once: break
        """id_N= int((len(gr['score']) + batch_size-1) / batch_size)
        ids = list(range(id_N))
        while True:
            if shuffle: random.shuffle(ids)#never swaped between batches...
            for i in ids:
                sl  = slice(i*batch_size, (i+1)*batch_size)
                grsl= dict()
                for name, value in gr.items(): grsl[name] = value[sl]
                grsl['se0'] = self.embed.map_jset(grsl['sj0'])
                grsl['se1'] = self.embed.map_jset(grsl['sj1'])
                x = [grsl['si0'],grsl['si1'],grsl['se0'],grsl['se1'],grsl['f0'],grsl['f1']]
                if self.c['target'] == 'classes':
                    y = grsl['classes']
                if self.c['target'] == 'score':
                    y = grsl['score1']
                yield (x,y)
                #yield grsl
            if once: break"""
    def predict(self, model, gr):
        batch_size = 16384#hardcoded
        ypred = []
        for grslpair,_ in self.sample_pairs(gr, batch_size, shuffle=False, once=True):
            if self.c['target'] == 'classes':
                ypred += list(model.predict(grslpair))#['classes'])
            if self.c['target'] == 'score':
                ypred += list(model.predict(grslpair))#['score1'])
                ypred = [y[0] for y in ypred]
        return np.array(ypred)
    def get_ypred_ygold(self, model, gr):
        if self.c['target'] == 'classes':
            ypred = loader.sts_categorical2labels(self.predict(model, gr))
        if self.c['target'] == 'score':
            ypred = np.dot(self.predict(model, gr),5.0)
        ygold = gr['score']#ygold = loader.sts_categorical2labels(gr['classes'])
        return ypred, ygold
    def task_eval_model(self, model, gr, fname, quiet=False, predout=''):
        ypred, ygold = self.get_ypred_ygold(model, gr)
        pr = pearsonr(ypred, ygold)[0]
        sr = spearmanr(ypred, ygold)[0]
        e  = mse(ypred, ygold)
        if predout!='':
            predname = predout+fname
            fpred=open(predname,"w")
            for ipred in range(len(ypred)):
                print("%f\t%f" %(ypred[ipred],ygold[ipred]),end="\t",file=fpred)
                print(" ".join(gr['s0'][ipred]),end="\t",file=fpred)
                print(" ".join(gr['s1'][ipred]),file=fpred)
                #print(" ".join(gr['s0'][ipred]).encode('utf-8'),end=",",file=fpred)
                #print(" ".join(gr['s1'][ipred]).encode('utf-8'),file=fpred)
        STSRes = namedtuple('STSRes', ['Pearson', 'Spearman', 'MSE'])
        if quiet: return STSRes(pr, sr, e)
        print('%s Pearson: %f' % (fname, pr,))
        print('%s Spearman: %f' % (fname, sr,))
        print('%s MSE: %f' % (fname, e,))
        return STSRes(pr, sr, e)
    def task_fit_model(self, model, gr, grv, grt, wfname, iwfname):
        print("",file=self.c['fplot'])
        #print(wfname,end="|",file=self.fplot)
        kwargs = dict()
        kwargs['generator'] = self.sample_pairs(gr, self.c['batch_size'])
        kwargs['samples_per_epoch'] = self.c['num_batchs']*self.c['batch_size']#int(len(gr['score'])*self.c['epoch_fract'])
        kwargs['nb_epoch']  = self.c['nb_epoch']
        class STSPearsonCB(Callback):
            def __init__(self, task, train_gr, valid_gr, test_gr, fout, cshow, wfname):
                self.best   = 0.0
                self.wfname = wfname
                self.nbatch = 0
                self.cshow  = cshow
                self.fout   = fout
                self.task   = task
                self.train_gr = train_gr
                self.valid_gr = valid_gr
                self.test_gr  = test_gr
            def on_batch_end(self, epoch, logs={}):
                if self.cshow:
                    prvl = self.task.task_eval_model(self.model, self.valid_gr, 'Valid', quiet=True).Pearson
                    if prvl > self.best:
                        self.best = prvl
                        self.task.model.save(self.wfname)
                    if self.nbatch%self.cshow==0:
                        restr = self.task.task_eval_model(self.model, self.train_gr, 'Train', quiet=True)
                        resvl = self.task.task_eval_model(self.model, self.valid_gr, 'Valid', quiet=True)
                        rests = self.task.task_eval_model(self.model, self.test_gr, 'Test', quiet=True)
                        prtr = restr.Pearson
                        prvl = resvl.Pearson
                        prts = rests.Pearson
                        print('                Pearson: train %f  valid %f  test %f' % (prtr, prvl, prts))
                        sptr = restr.Spearman
                        spvl = resvl.Spearman
                        spts = rests.Spearman
                        print('               Spearman: train %f  valid %f  test %f' % (sptr, spvl, spts))
                        metr = restr.MSE
                        mevl = resvl.MSE
                        mets = rests.MSE
                        print('                    MSE: train %f  valid %f  test %f' % (metr, mevl, mets))
                        print("%f,%f,%f" %(prtr, prvl, prts), end="|", file=self.fout)
                        self.fout.flush()
                self.nbatch += 1
            def on_epoch_end(self, epoch, logs={}):
                restr = self.task.task_eval_model(self.model, self.train_gr, 'Train', quiet=True)
                resvl = self.task.task_eval_model(self.model, self.valid_gr, 'Valid', quiet=True)
                rests = self.task.task_eval_model(self.model, self.test_gr, 'Test', quiet=True)
                prtr = restr.Pearson
                prvl = resvl.Pearson
                prts = rests.Pearson
                print('                Pearson: train %f  valid %f  test %f' % (prtr, prvl, prts))
                sptr = restr.Spearman
                spvl = resvl.Spearman
                spts = rests.Spearman
                print('               Spearman: train %f  valid %f  test %f' % (sptr, spvl, spts))
                metr = restr.MSE
                mevl = resvl.MSE
                mets = rests.MSE
                print('                    MSE: train %f  valid %f  test %f' % (metr, mevl, mets))
                if not self.cshow:
                    print("%f,%f,%f" %(prtr, prvl, prts), end="|", file=self.fout)
                    self.fout.flush()
                    prvl = self.task.task_eval_model(self.model, self.valid_gr, 'Valid', quiet=True).Pearson
                    if prvl > self.best:
                        self.best = prvl
                        self.task.model.save(self.wfname)
                logs['pearson'] = prvl
        kwargs['callbacks'] = [STSPearsonCB(self, gr, grv, grt, self.c['fplot'], self.c['showbatchs'], wfname),
            #ModelCheckpoint(wfname, save_best_only=True, monitor='pearson', mode='max'),
            EarlyStopping(monitor='pearson', mode='max', patience=self.c['patience'])]
        return model.fit_generator(verbose=self.c['verbose'],**kwargs)

def task():
    return STSTask()
