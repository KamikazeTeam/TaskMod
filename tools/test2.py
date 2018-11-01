import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf 
from random import shuffle
import tqdm

acc = 0.005
X = np.arange(acc, 1, acc)
Z = (-np.cos(X*10)+1.0)/4.0+X/2.0
data_train = []
for i in range(len(Z)):
    data_train.append((X[i],Z[i]))
#shuffle(data_train)
x_train = np.array([[data[0]] for data in data_train])
y_train = np.array([[data[1]] for data in data_train])

tfconfig = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=tfconfig)
x_input = tf.placeholder(tf.float32, shape=x_train.shape, name='x_input')
y_input = tf.placeholder(tf.float32, name='y_input') 
initializer = None#tf.random_uniform() #tf.constant_initializer(1.0)
m_value = tf.layers.dense(inputs=x_input, units=2, activation=tf.nn.relu,
        kernel_initializer=initializer, bias_initializer=tf.constant_initializer(1.0))
y_value = tf.layers.dense(inputs=m_value, units=1, activation=None,#tf.nn.softmax, 
        kernel_initializer=initializer, bias_initializer=tf.constant_initializer(1.0))
print(len(tf.trainable_variables()))
w_value = [v for v in tf.trainable_variables()][0]
b_value = [v for v in tf.trainable_variables()][1]
w2_value = [v for v in tf.trainable_variables()][2]
b2_value = [v for v in tf.trainable_variables()][3]
loss_op = tf.reduce_mean(tf.pow(y_input-y_value, 2))
train_op= tf.train.AdamOptimizer(0.1).minimize(loss_op) # AdamOptimizer GradientDescentOptimizer
gradw_op= tf.gradients(loss_op, w_value) 
gradb_op= tf.gradients(loss_op, b_value) 
init = tf.global_variables_initializer() 
sess.run(init)

lossalls = []
y_preds  = []
for it in tqdm.tqdm(range(10000)): 
    i = it%len(x_train)
    y, loss = sess.run([y_value, loss_op], feed_dict={x_input: x_train, y_input: y_train})
    y_predsarray=np.array(y)
    lossalls.append(loss)
    if 0:#it%2000==0:
        w, b, gradw, gradb = sess.run([w_value, b_value, gradw_op, gradb_op], feed_dict={x_input: x_train, y_input: y_train})
        w2, b2 = sess.run([w2_value, b2_value], feed_dict={x_input: x_train, y_input: y_train})
        print('w',w)
        print('b',b)
        print('w2',w2)
        print('b2',b2)
        print("epoch: {} \t x: {} \t xr:  \t yt: {}".format(it, x_train[i], y_train[i])) #, x_train[i].reshape((2,1)), y_train[i])) 
        print("epoch: {} \t w: {} \t b: {} \t y: {} \t gradw: {} \t gradb: {} \t loss: {}".format(it, w, b, y, gradw, gradb, loss))
        print('')
    sess.run([train_op], feed_dict={x_input: x_train, y_input: y_train})

start = 0
plt.plot(np.arange(start, len(lossalls), 1),lossalls[start:],color='r',alpha=0.5)
Z0loss = ((Z-X)**2).mean(axis=None)
plt.axhline(y=Z0loss, linewidth=0.5, color='k')
plt.axhline(y=0.03, linewidth=0.5, color='k')
plt.axhline(y=0.02, linewidth=0.5, color='k')
plt.axhline(y=0.01, linewidth=0.5, color='k')
plt.axhline(y=0.008, linewidth=0.5, color='k')
plt.axhline(y=0.006, linewidth=0.5, color='k')
plt.axhline(y=0.004, linewidth=0.5, color='k')
plt.axhline(y=0.002, linewidth=0.5, color='k')
plt.axhline(y=0.00, linewidth=0.5, color='k')
plt.ylim([-0.01,0.06])
print(Z0loss)
print(lossalls[:10])
print(lossalls[-20:])

plt.figure(2)
plt.plot(X,Z,color='r',alpha=0.5)
print(y_predsarray.shape)
y = y_predsarray.reshape(Z.shape)
plt.plot(X,y,color='b',alpha=0.5)
#plt.ylim([0.0,1.0])

if 0:
    m = sess.run([m_value], feed_dict={x_input: x_train, y_input: y_train})
    w2, b2 = sess.run([w2_value, b2_value], feed_dict={x_input: x_train, y_input: y_train})

    wmin,wmax = np.amin(w2),np.amax(w2)
    print(wmin,wmax)
    numbs = 2
    wacc = 0.5#(wmax-wmin)/numbs
    w21l = np.arange(wmin-10, wmax+10, wacc)
    w22l = np.arange(wmin-10, wmax+10, wacc)
    w23l = np.arange(wmin-10, wmax+10, wacc)
    w22,w23 = np.meshgrid(w22l, w23l)
    w2l = []
    for w21e in w21l:
        for w22e in w22l:
            for w23e in w23l:
                w2l.append([[w21e],[w22e],[w23e]])
    lossl = []
    for e in tqdm.tqdm(range(len(w2l))):
        w2e = np.array(w2l[e])
        ye = np.dot(m[0],w2e)+b2
        losse = pow((ye-y_train),2).mean(axis=None)
        lossl.append(losse)
    for i in range(len(w21l)):
        if i%(int(len(w21l)/5))!=0: 
            continue
        lenth = len(w22l)*len(w23l)
        print(w2l[i*lenth])
        losslc= lossl[i*lenth:(i+1)*lenth]
        losslc=np.array(losslc)
        print(losslc.shape)
        print(w22.shape)
        losslc=losslc.reshape(w22.shape)
        print(losslc.shape)
        print(w22.shape)
        axmap = Axes3D(plt.figure())
        axmap.plot_surface(w22, w23, losslc, rstride=1, cstride=1, cmap='rainbow', alpha=0.8) 



plt.show()
sess.close()
