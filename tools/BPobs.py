import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf 
from random import shuffle
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
        for units in unitslist[1:len(unitslist)-1]:
            #h_valueo= tf.identity(h_value)
            h_value = tf.layers.dense(inputs=h_value, units=units, activation=tf.nn.relu, kernel_initializer=initializer, bias_initializer=initializer)
            h_value = tf.layers.batch_normalization(inputs=h_value,scale=False,training=True)#tf.contrib.layers.batch_norm(h_value)
            #h_value = tf.nn.relu(h_value)
            #h_value = tf.layers.dense(inputs=h_value, units=units, activation=tf.nn.relu, kernel_initializer=initializer, bias_initializer=initializer)
            #h_value = tf.contrib.layers.batch_norm(h_value)#tf.layers.batch_normalization
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
def figlossandpred(lossalls, y_preds, X, Y, Z, expdir, randseed, maxstep, outitvl, ylines, ylim):
    plt.figure(11)
    start = 0
    plt.plot(np.arange(start, len(lossalls), 1),lossalls[start:],color='r',alpha=0.5)
    for yline in ylines:
        plt.axhline(y=yline, linewidth=0.5, color='k')
    plt.ylim([0.0,ylim])
    for i in range(outitvl):
        xline = int(maxstep/outitvl)*(i+1)
        plt.axvline(x=xline, linewidth=0.5, color='k')
    plt.savefig(expdir+'loss'+str(randseed)+".png", figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
    axp = Axes3D(plt.figure(12))
    Zp  = np.array(y_preds).reshape(Z.shape)
    axp.plot_surface(X,Y,Zp, rstride=1, cstride=1, cmap='rainbow', alpha=0.5) 
    plt.savefig(expdir+'pred'+str(randseed)+".png", figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
def setsession(randseed):
    tfconfig = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=tfconfig)
    random.seed(randseed)
    np.random.seed(randseed)
    tf.set_random_seed(randseed)
    return sess
def targetfunction(acc,lim,freq,randflag):
    Xl = np.arange(acc, lim, acc)
    Yl = np.arange(acc, lim, acc)
    X, Y = np.meshgrid(Xl, Yl)
    R = np.sqrt((X-lim/2)**2+(Y-lim/2)**2)
    Z = (np.sin(R*freq)+1.0)/4.0+0.25
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
    acc, lim, freq, randflag = 0.025, 2.0, 10.0, False
    X,Y,Z,x_train,y_train = targetfunction(acc=acc,lim=lim,freq=freq,randflag=randflag)

    randseed = int(sys.argv[1])
    sess = setsession(randseed)

    unitslist= [2,4,1]
    model = Model(x_train,y_train,unitslist)

    dt, maxstep, outitvl = 0.1, 1000, 3
    expdir = str(acc).replace('.','-')+'+'+str(unitslist).replace('[','-').replace(']','-')+'+'+str(dt).replace('.','-')+'+'+str(maxstep)+'/'
    if not os.path.exists(expdir): os.makedirs(expdir)
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
        ylines, ylim = [0.01,0.005,0.004,0.003,0.002,0.001,0.0005], 0.02
        figlossandpred(lossalls, y_preds, X, Y, Z, expdir, randseed, maxstep, outitvl, ylines, ylim)

    print(sys.argv[1],':',sys.argv[2])
    if sys.argv[1]==sys.argv[2] or int(sys.argv[1])%100==0:
        plt.figure(111)
        plt.hist(lossmins, bins=np.arange(0, max(lossmins)+0.0001, 0.0001), normed=False,facecolor='r',edgecolor='r',hold=0,alpha=0.2)
        plt.savefig(expdir+'lossmins'+".png", figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
        ax = Axes3D(plt.figure(112))
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.2)
        plt.savefig(expdir+'gold'+".png", figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    main()
