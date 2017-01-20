from keras import backend as K
from keras.models import Model
from keras.layers import merge, Dense
from keras.regularizers import l2

from mods.layers import *

class AbstractMod():
    def __init__(self):
        self.testlayers = []

    def scorer(self, *args):
        return self.scorer_multilayer(*args)
        return self.scorer_multiperspective(*args)

    def scorer_multilayer(self, tensor0, tensor1, output_dim, dimoflayers, l2reg, act, init):
        absdiff = merge([tensor0,tensor1], mode=lambda X:K.abs(X[0] - X[1]), output_shape=lambda X: X[0], name='absdiff')
        muldiff = merge([tensor0,tensor1], mode='mul', name='muldiff')
        alldiff = merge([absdiff,muldiff], mode='concat', name='alldiff')
        mlpdiff = alldiff
        for i, D in enumerate(dimoflayers):
            dendiff = Dense(name='mlp'+'hdn%d'%(i,), 
                            output_dim=int(D), 
                            W_regularizer=l2(l2reg), 
                            activation=act, 
                            init=init)(mlpdiff)
            mlpdiff = dendiff
        outputfinal = Dense(name='outputfinal', 
                            output_dim=output_dim,
                            W_regularizer=l2(l2reg), 
                            activation='softmax', 
                            init=init)(mlpdiff)
        return outputfinal

    def scorer_multiperspective(self, tensor0, tensor1, output_dim, *args):
        conf = dict()
        conf['num_filters_A'] = 200
        conf['num_filters_B'] = 20
        conf['n_hidden'] = 150
        conf['lambda'] = 1e-4#regularization parameter
        import sys
        wss = [1, 2, 3, sys.maxsize]

        sentences = [tensor0, tensor1]

        atten_embeds  = AttentionInputLayer(sentences).atten_embeds
        setence_model = SentenceModelingLayer(conf, atten_embeds, wss)
        #fea_v = setence_model.vertical_comparison()
        fea_h = setence_model.horizontal_comparison()
        ss = SimilarityScoreLayer(fea_h, conf)
        p  = ss.generate_p()

        return p

    def debug_model(self, mergedinp0, mergedinp1, N, c, output0, output1,
                    inputsint0, inputsint1, inputsemb0, inputsemb1, inputsnlp0, inputsnlp1, outputfinal, sample_pairs, grt):
        for x,_ in sample_pairs(grt, batch_size=1, shuffle=False, once=True):
            model = Model(input=[inputsint0,inputsint1,inputsemb0,inputsemb1,inputsnlp0,inputsnlp1],
                          output=[mergedinp0])
            value = model.predict(x)
            print("mergedinp0")
            print(value.shape)
            print(value)
            model = Model(input=[inputsint0,inputsint1,inputsemb0,inputsemb1,inputsnlp0,inputsnlp1],
                          output=[output0])
            value = model.predict(x)
            print("output0")
            print(value.shape)
            print(value)
            model = Model(input=[inputsint0,inputsint1,inputsemb0,inputsemb1,inputsnlp0,inputsnlp1],
                          output=[outputfinal])
            value = model.predict(x)
            print("outputfinal")
            print(value.shape)
            print(value)
            for testlayer in self.testlayers:
                model = Model(input=[inputsint0,inputsint1,inputsemb0,inputsemb1,inputsnlp0,inputsnlp1],
                              output=[testlayer])
                value = model.predict(x)
                print(testlayer)
                print(value.shape)
                print(value)
            print('debug end')
            exit()
        exit()

