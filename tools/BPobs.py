import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf 
from random import shuffle
from sklearn.preprocessing import scale,normalize
import tqdm, sys, os, warnings, random
warnings.filterwarnings('ignore')
###########################################################################
class Model:
    def __init__(self,x_train,y_train,unitslist):
        self.unitslist = unitslist
        self.x_input = tf.placeholder(tf.float32, shape=x_train.shape, name='x_input')
        self.y_input = tf.placeholder(tf.float32, name='y_input')
        initializer = None#tf.random_uniform() #tf.constant_initializer(1.0)
        self.h_values = []
        h_value = self.x_input
        self.h_values.append(h_value)
        for i,units in enumerate(unitslist[1:len(unitslist)-1]):
            if i==0:
                h_value = tf.layers.dense(inputs=h_value, units=units, activation=tf.nn.relu, kernel_initializer=initializer, bias_initializer=initializer)
                h_value = tf.layers.batch_normalization(inputs=h_value,scale=False,training=True)#tf.contrib.layers.batch_norm(h_value)
            #h_valueo= tf.identity(h_value)
            h_value = tf.layers.dense(inputs=h_value, units=units, activation=tf.nn.relu, kernel_initializer=initializer, bias_initializer=initializer)
            h_value = tf.layers.batch_normalization(inputs=h_value,scale=False,training=True)#tf.contrib.layers.batch_norm(h_value)
            #h_value = tf.nn.relu(h_value)
            h_value = tf.layers.dense(inputs=h_value, units=units, activation=tf.nn.relu, kernel_initializer=initializer, bias_initializer=initializer)
            h_value = tf.layers.batch_normalization(inputs=h_value,scale=False,training=True)
            #h_value = tf.nn.relu(h_value)
            #h_value = h_value + h_valueo
            self.h_values.append(h_value)
        self.y_value = tf.layers.dense(inputs=h_value, units=1, activation=None, kernel_initializer=initializer, bias_initializer=initializer)#tf.nn.softmax
        self.h_values.append(self.y_value)
        self.loss_op  = tf.reduce_mean(tf.pow(self.y_input-self.y_value, 2))
        print(len(tf.trainable_variables()))
        if 0:
            self.w_values = []
            self.b_values = []
            for i in range(0,len(tf.trainable_variables()),2):
                w_value = [v for v in tf.trainable_variables()][i]
                b_value = [v for v in tf.trainable_variables()][i+1]
                self.w_values.append(w_value)
                self.b_values.append(b_value)
            self.gradh_ops= []
            for h_value in self.h_values:
                gradh_op = tf.gradients(self.loss_op, h_value)
                self.gradh_ops.append(gradh_op)
            self.gradw_ops= []
            for w_value in self.w_values:
                gradw_op = tf.gradients(self.loss_op, w_value)
                self.gradw_ops.append(gradw_op)
            self.gradb_ops= []
            for b_value in self.b_values:
                gradb_op = tf.gradients(self.loss_op, b_value)
                self.gradb_ops.append(gradb_op)
###########################################################################

###########################################################################
class Trainer:
    def __init__(self,x_train,y_train,sess,model,expdir):
        self.x_train, self.y_train, self.sess, self.model, self.expdir = x_train, y_train, sess, model, expdir
    def fit(self,dt,maxstep,outitvl):
        optimizer= tf.train.AdamOptimizer(dt)# AdamOptimizer GradientDescentOptimizer
        train_op = optimizer.minimize(self.model.loss_op)
        self.sess.run(tf.global_variables_initializer())
        lossalls = []
        for it in tqdm.tqdm(range(maxstep)):
            loss, _ = self.sess.run([self.model.loss_op, train_op], feed_dict={self.model.x_input: self.x_train, self.model.y_input: self.y_train})
            lossalls.append(loss)
        y_preds = self.sess.run([self.model.y_value], feed_dict={self.model.x_input: self.x_train, self.model.y_input: self.y_train})
        self.sess.close()
        return lossalls, y_preds
###########################################################################
def figlossandpred(lossalls, y_preds, X, Y, Z, expdir, randseed, maxstep, outitvl, ylines, ylimmin, ylimmax):
    plt.figure(11)
    start = 0
    plt.plot(np.arange(start, len(lossalls), 1),lossalls[start:],color='r',alpha=0.5)
    for yline in ylines:
        plt.axhline(y=yline, linewidth=0.5, color='k')
    plt.yscale('log')
    plt.ylim([ylimmin,ylimmax])
    for i in range(outitvl):
        xline = int(maxstep/outitvl)*(i+1)
        plt.axvline(x=xline, linewidth=0.5, color='k')
    plt.savefig(expdir+'loss'+str(randseed)+".png", figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
    axp = Axes3D(plt.figure(12))
    Zp  = np.array(y_preds).reshape(Z.shape)
    axp.plot_surface(X,Y,Zp, rstride=1, cstride=1, cmap='rainbow', alpha=0.6) 
    plt.savefig(expdir+'pred'+str(randseed)+".png", figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
def setsession(randseed):
    tfconfig = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=tfconfig)
    random.seed(randseed)
    np.random.seed(randseed)
    tf.set_random_seed(randseed)
    return sess
def targetfunction(acc,lim,cnum,freq,randflag):
    Xl = sorted(np.random.uniform(low=-lim, high=lim, size=(acc,)))#np.linspace(-lim, lim, acc)
    Yl = sorted(np.random.uniform(low=-lim, high=lim, size=(acc,)))
    Xl = scale(Xl, axis=0, with_mean=True,with_std=True,copy=True)#Xl / np.linalg.norm(Xl)
    Yl = scale(Yl, axis=0, with_mean=True,with_std=True,copy=True)
    X, Y = np.meshgrid(Xl, Yl)
    Rijs, Zijs = [], []
    xc = np.random.uniform(low=-lim, high=lim, size=(cnum,))
    yc = np.random.uniform(low=-lim, high=lim, size=(cnum,))
    xc = scale(xc, axis=0, with_mean=True,with_std=True,copy=True)
    yc = scale(yc, axis=0, with_mean=True,with_std=True,copy=True)
    for i in range(cnum):
        Rij = np.sqrt((X-xc[i])**2+(Y-yc[i])**2)
        Rijs.append(Rij)
    for Rij in Rijs:
        Zij = np.sin(Rij*freq)
        Zijs.append(Zij)
    Z  = np.array(Zijs).max(axis=0)######
    Z = scale(Z, axis=0, with_mean=True,with_std=True,copy=True)
    #R = np.sqrt((X-lim/2)**2+(Y-lim/2)**2)
    #Z = (np.sin(R*freq)+1.0)/4.0+0.25
    data_train = []
    for i in range(len(Z)):
        for j in range(len(Z[i])):
            data_train.append((X[i][j],Y[i][j],Z[i][j]))
    if randflag: shuffle(data_train)
    x_train = np.array([[data[0],data[1]] for data in data_train])
    y_train = np.array([[data[2]] for data in data_train])
    print('Z0loss:', ((Z)**2).mean(axis=None))
    return X,Y,Z,x_train,y_train
###########################################################################
def main():
    randseed = int(sys.argv[1])
    sess = setsession(randseed)
    sep = 'T'
    sysargv3 = sys.argv[3].replace('|',sep)
    hyps = sysargv3.split(sep)
    modelname = hyps[0]
    acc, lim, cnum, freq, randflag = int(hyps[1]), 1.0, int(hyps[2]), 10.0, False
    unitslist= [int(num) for num in hyps[3].split('-')]
    dt, maxstep, outitvl = float(hyps[4]), int(hyps[5]), 3
    limmin, limmax, nylines, nlogbins = 1e-5, 5e-2, 10, 50
    print(sys.argv[1],':',sys.argv[2],':',sys.argv[3])
    expdir = sysargv3+'/'
    if not os.path.exists(expdir): os.makedirs(expdir)

    X,Y,Z,x_train,y_train = targetfunction(acc=acc,lim=lim,cnum=cnum,freq=freq,randflag=randflag)
    model = Model(x_train,y_train,unitslist)
    trainer = Trainer(x_train,y_train,sess,model,expdir)
    lossalls, y_preds = trainer.fit(dt,maxstep,outitvl)

    flossmins = open(expdir+'lossmins','a')
    print(min(lossalls),end=',',file=flossmins)
    flossmins.close()
    flossmins = open(expdir+'lossmins','r')
    lossmins = flossmins.read().splitlines()[0].split(",")[:-1]
    lossmins = [float(lossmin) for lossmin in lossmins]
    flossmins.close()
    if lossmins[-1]<=min(lossmins) or len(lossmins)<10:
        ylines, ylimmin, ylimmax = np.logspace(np.log10(limmin),np.log10(limmax),nylines), limmin, limmax
        figlossandpred(lossalls, y_preds, X, Y, Z, expdir, randseed, maxstep, outitvl, ylines, ylimmin, ylimmax)
    if sys.argv[1]==sys.argv[2]:# or int(sys.argv[1])%100==0:
        plt.figure(111)
        xlimmin, xlimmax = min(lossmins+[limmin]) ,max(lossmins+[limmax])
        logbins = np.logspace(np.log10(xlimmin),np.log10(xlimmax),nlogbins) #bins = np.arange(xlimmin, xlimmax, 1e-6) #linspace
        plt.hist(lossmins, bins=logbins, normed=False,facecolor='r',edgecolor='r',hold=0,alpha=0.2)
        plt.xscale('log')
        numparas = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        figname = sysargv3+sep+'lossmins'+sep+str(x_train.shape[0])+'+'+str(numparas)+sep+str(min(lossmins))+'+'+str(np.median(lossmins))+".png"
        plt.savefig(figname, figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
        fresults=open('dohypersresults','a')
        print(sys.argv[3]+','+str(x_train.shape[0])+','+str(numparas)+','+str(min(lossmins))+','+str(np.median(lossmins)),file=fresults)
        fresults.close()###############
        goldname = str(acc)+'+'+str(lim)+'+'+str(cnum)+'+'+str(freq)+'gold'+".png"
        if not os.path.exists(goldname):
            ax = Axes3D(plt.figure(112))
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.8)
            plt.savefig(goldname, figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
            #plt.show()

if __name__ == '__main__':
    main()
