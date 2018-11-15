import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf 
from random import shuffle
import tqdm, sys, os, warnings
warnings.filterwarnings('ignore')

acc = 0.025
Xl = np.arange(acc, 1, acc)
Yl = np.arange(acc, 1, acc)
X, Y = np.meshgrid(Xl, Yl)
R = np.sqrt((X-0.5)**2+(Y-0.5)**2)
Z = (np.sin(R*10)+1.0)/4.0+0.25
data_train = []
for i in range(len(Z)):
    for j in range(len(Z[i])):
        data_train.append((X[i][j],Y[i][j],Z[i][j]))
#shuffle(data_train)
x_train = np.array([[data[0],data[1]] for data in data_train])
y_train = np.array([[data[2]] for data in data_train])

tfconfig = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=tfconfig)
tf.set_random_seed(int(sys.argv[1]))
x_input = tf.placeholder(tf.float32, shape=x_train.shape, name='x_input')
y_input = tf.placeholder(tf.float32, name='y_input')
initializer = None#tf.random_uniform() #tf.constant_initializer(1.0)

unitslist= [2,8,8,8,8,8,1]
h_values = []
h_value = x_input
h_values.append(h_value)
for units in unitslist[1:len(unitslist)-1]:
    h_value = tf.layers.dense(inputs=h_value, units=units, activation=tf.nn.relu, kernel_initializer=initializer, bias_initializer=initializer)
    h_values.append(h_value)
y_value = tf.layers.dense(inputs=h_value, units=1, activation=None, kernel_initializer=initializer, bias_initializer=initializer)#tf.nn.softmax
h_values.append(y_value)
print(len(tf.trainable_variables()))
w_values = []
b_values = []
for i in range(0,len(tf.trainable_variables()),2):
    w_value = [v for v in tf.trainable_variables()][i]
    b_value = [v for v in tf.trainable_variables()][i+1]
    w_values.append(w_value)
    b_values.append(b_value)
loss_op  = tf.reduce_mean(tf.pow(y_input-y_value, 2))
dt       = 0.1
opt      = tf.train.AdamOptimizer(dt)# AdamOptimizer GradientDescentOptimizer
train_op = opt.minimize(loss_op)
gradh_ops= []
for h_value in h_values:
    gradh_op = tf.gradients(loss_op, h_value)
    gradh_ops.append(gradh_op)
gradw_ops= []
for w_value in w_values:
    gradw_op = tf.gradients(loss_op, w_value)
    gradw_ops.append(gradw_op)
gradb_ops= []
for b_value in b_values:
    gradb_op = tf.gradients(loss_op, b_value)
    gradb_ops.append(gradb_op)
init = tf.global_variables_initializer() 
sess.run(init) 

maxstep  = 1000
outitvl  = 3
expdir   = str(acc)+':'+str(unitslist)+':'+str(dt)+':'+str(maxstep)+'/'
if not os.path.exists(expdir): os.makedirs(expdir)
lossalls = []
y_preds  = []
for it in tqdm.tqdm(range(maxstep)): 
    #loss = sess.run([loss_op], feed_dict={x_input: x_train, y_input: y_train})[0]
    y, loss = sess.run([y_value, loss_op], feed_dict={x_input: x_train, y_input: y_train})
    lossalls.append(loss)
    if 0:#it%int(maxstep/outitvl)==0:
        if 1:
            hs = []
            for h_value in h_values:
                h = sess.run([h_value], feed_dict={x_input: x_train, y_input: y_train})
                hs.append(h)
            ws = []
            for w_value in w_values:
                w = sess.run([w_value], feed_dict={x_input: x_train, y_input: y_train})
                ws.append(w)
            bs = []
            for b_value in b_values:
                b = sess.run([b_value], feed_dict={x_input: x_train, y_input: y_train})
                bs.append(b)
        if 0:
            gradhs = []
            for gradh_op in gradh_ops:
                gradh = sess.run([gradh_op], feed_dict={x_input: x_train, y_input: y_train})
                gradhs.append(gradh)
            gradws = []
            for gradw_op in gradw_ops:
                gradw = sess.run([gradw_op], feed_dict={x_input: x_train, y_input: y_train})
                gradws.append(gradw)
            gradbs = []
            for gradb_op in gradb_ops:
                gradb = sess.run([gradb_op], feed_dict={x_input: x_train, y_input: y_train})
                gradbs.append(gradb)
        if 0:#it==maxstep-1:
            print('hs',hs)
            print('ws',ws)
            print('bs',bs)
            print('gradhs',gradhs)
            print('gradws',gradws)
            print('gradbs',gradbs)
        if 1:#it==maxstep-1:
            plt.figure(1)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.suptitle('hs')
            plt.rc('xtick', labelsize=3)
            plt.rc('ytick', labelsize=3)
            for i,h in enumerate(hs):
                for fn in range(unitslist[i]):
                    datafn = [data[fn] for data in h[0]]
                    plt.subplot(len(hs),max(unitslist),i*max(unitslist)+fn+1)
                    plt.hist(datafn, bins=np.arange(min(datafn)-0.1, max(datafn)+0.1, (max(datafn)-min(datafn)+0.2)/20.0), normed=False,facecolor='r',edgecolor='r',hold=0,alpha=0.2)
            plt.savefig(expdir+'hs'+str(sys.argv[1])+'-'+str(int(it/int(maxstep/outitvl)))+".png", figsize=(64, 36), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
            plt.figure(2)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.suptitle('bs+ws')
            for i,w in enumerate(ws):
                for fn in range(unitslist[i]):
                    datafn = w[0][fn]#[data[fn] for data in w[0]]
                    plt.subplot(len(ws)+1,max(unitslist)+1,i*(max(unitslist)+1)+fn+1+1)
                    plt.hist(datafn, bins=np.arange(min(datafn)-0.9, max(datafn)+0.9, (max(datafn)-min(datafn)+1.8)/20.0), normed=False,facecolor='r',edgecolor='r',hold=0,alpha=0.2)
                datafn = bs[i][0]#w[0][fn]#[data[fn] for data in w[0]]
                plt.subplot(len(ws)+1,max(unitslist)+1,i*(max(unitslist)+1)+1)
                plt.hist(datafn, bins=np.arange(min(datafn)-0.9, max(datafn)+0.9, (max(datafn)-min(datafn)+1.8)/20.0), normed=False,facecolor='r',edgecolor='r',hold=0,alpha=0.2)
            plt.savefig(expdir+'bs+ws'+str(sys.argv[1])+'-'+str(int(it/int(maxstep/outitvl)))+".png", figsize=(64, 36), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
        axp = Axes3D(plt.figure(3))
        Zp  = np.array(y).reshape(Z.shape)
        axp.plot_surface(X,Y,Zp, rstride=1, cstride=1, cmap='rainbow', alpha=0.8) 
        plt.savefig(expdir+'pred'+str(sys.argv[1])+'-'+str(int(it/int(maxstep/outitvl)))+".png", figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)

    sess.run([train_op], feed_dict={x_input: x_train, y_input: y_train})

flossmins = open(expdir+'lossmins','a')
print(min(lossalls),end=',',file=flossmins)
flossmins.close()
flossmins = open(expdir+'lossmins','r')
lossmins = [float(lossmin) for lossmin in flossmins.read().splitlines()[0].split(",")[:-1]]
flossmins.close()
#print(lossmins)
#print(min(lossalls))
#print(min(lossmins))
if lossmins[-1]<=min(lossmins):
    #print('in')
    Z0loss = ((Z)**2).mean(axis=None)
    #print(Z0loss)
    #print(lossalls[:10])
    #print(lossalls[-20:])
    plt.figure(11)
    start = 0
    plt.plot(np.arange(start, len(lossalls), 1),lossalls[start:],color='r',alpha=0.5)
    plt.axhline(y=Z0loss, linewidth=0.5, color='k')
    plt.axhline(y=0.03, linewidth=0.5, color='k')
    plt.axhline(y=0.02, linewidth=0.5, color='k')
    plt.axhline(y=0.01, linewidth=0.5, color='k')
    plt.axhline(y=0.008, linewidth=0.5, color='k')
    plt.axhline(y=0.006, linewidth=0.5, color='k')
    plt.axhline(y=0.004, linewidth=0.5, color='k')
    plt.axhline(y=0.002, linewidth=0.5, color='k')
    plt.axhline(y=0.00, linewidth=0.5, color='k')
    plt.ylim([0.0,0.04])
    for i in range(outitvl):
        plt.axvline(x=int(maxstep/outitvl)*(i+1), linewidth=0.5, color='k')
    plt.savefig(expdir+'loss'+str(sys.argv[1])+".png", figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
    #ax  = Axes3D(plt.figure(12))
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.2) 
    axp = Axes3D(plt.figure(13))
    Zp  = np.array(y).reshape(Z.shape)
    axp.plot_surface(X,Y,Zp, rstride=1, cstride=1, cmap='rainbow', alpha=0.8) 
    plt.savefig(expdir+'pred'+str(sys.argv[1])+".png", figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)

#plt.show()
sess.close()

print(sys.argv[1],':',sys.argv[2])
if sys.argv[1]==sys.argv[2] or int(sys.argv[1])%100==0:
    lossmins = open(expdir+'lossmins','r').read().splitlines()[0].split(",")[:-1]
    lossmins = [float(lossmin) for lossmin in lossmins]
    plt.figure(111)
    plt.hist(lossmins, bins=np.arange(0, max(lossmins)+0.001, 0.001), normed=False,facecolor='r',edgecolor='r',hold=0,alpha=0.2)
    plt.savefig(expdir+'lossmins'+".png", figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
