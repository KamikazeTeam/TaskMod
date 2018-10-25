import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf 
from random import shuffle
import tqdm

Xl = np.arange(0.1, 1, 0.1)
Yl = np.arange(0.1, 1, 0.1)
X, Y = np.meshgrid(Xl, Yl)
R = np.sqrt((X-0.5)**2+(Y-0.5)**2)
Z = (np.sin(R*10)+1.0)/4.0+0.25
#print(X.shape)
#print(Y.shape)
#print(Z.shape)
#print(X[:3])
#print(Y[:3])
#print(Z[:3])
#ax  = Axes3D(plt.figure())
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 
#plt.show()

data_train = []
for i in range(len(Z)):
    for j in range(len(Z[i])):
        data_train.append((X[i][j],Y[i][j],Z[i][j]))
#shuffle(data_train)
x_train = np.array([[data[0],data[1]] for data in data_train])
y_train = np.array([[data[2]] for data in data_train])
#print(x_train.shape)
#print(y_train.shape)
#print(x_train[:3])
#print(y_train[:3])

tfconfig = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=tfconfig)
x_input = tf.placeholder(tf.float32, shape=(1,2), name='x_input') 
y_input = tf.placeholder(tf.float32, name='y_input') 
#w = tf.Variable(2.0, name='weight') 
#b = tf.Variable(1.0, name='biases') 
#y = tf.add(tf.multiply(x_input, w), b)
y_value = tf.layers.dense(inputs=x_input, units=1, activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(1.0), bias_initializer=tf.constant_initializer(1.0))
#y_value = tf.layers.dense(inputs=x_input, units=1, activation=None,
#        kernel_initializer=tf.constant_initializer(2.0), bias_initializer=tf.constant_initializer(1.0))
#y_value = tf.layers.dense(inputs=x_input, units=4, activation=tf.nn.softmax, 
#        kernel_initializer=tf.constant_initializer(2.0), bias_initializer=tf.constant_initializer(1.0))
#y_value = tf.layers.dense(inputs=y_value, units=16, activation=tf.nn.relu, 
#        kernel_initializer=tf.constant_initializer(1.0), bias_initializer=tf.constant_initializer(1.0))
y_value = tf.layers.dense(inputs=y_value, units=1, activation=tf.nn.relu, 
        kernel_initializer=tf.constant_initializer(1.0), bias_initializer=tf.constant_initializer(1.0))
y_value = tf.layers.dense(inputs=y_value, units=1, activation=tf.nn.relu, 
        kernel_initializer=tf.constant_initializer(1.0), bias_initializer=tf.constant_initializer(1.0))
y_value = tf.layers.dense(inputs=y_value, units=1, activation=tf.nn.relu,#tf.nn.softmax, 
        kernel_initializer=tf.constant_initializer(1.0), bias_initializer=tf.constant_initializer(1.0))
print(len(tf.trainable_variables()))
w_value = [v for v in tf.trainable_variables()][0]
b_value = [v for v in tf.trainable_variables()][1]
loss_op = tf.reduce_mean(tf.pow(y_input-y_value, 2))
train_op= tf.train.GradientDescentOptimizer(0.0001).minimize(loss_op) 
gradw_op= tf.gradients(loss_op, w_value) 
gradb_op= tf.gradients(loss_op, b_value) 
init = tf.global_variables_initializer() 
sess.run(init) 

lossalls = []
lossis   = []
y_preds  = []
for it in tqdm.tqdm(range(1000)): 
    i = it%len(x_train)
    #_, w, b, y, grads, loss = sess.run([train_op, w_value, b_value, y_value, grads_op, loss_op], feed_dict={x_input: x_train[i].reshape((1,2)), y_input: y_train[i]})
    w, b, y, gradw, gradb, loss = sess.run([w_value, b_value, y_value, gradw_op, gradb_op, loss_op], feed_dict={x_input: x_train[i].reshape((1,2)), y_input: y_train[i]})
    lossref  = pow((y-y_train[i]),2).mean(axis=None)
    gradref0 = 2*(w[0]*x_train[i][0]+w[1]*x_train[i][1]+b-y_train[i])*x_train[i][0]
    gradref1 = 2*(w[0]*x_train[i][0]+w[1]*x_train[i][1]+b-y_train[i])*x_train[i][1]
    gradrefb = 2*(w[0]*x_train[i][0]+w[1]*x_train[i][1]+b-y_train[i])
    tolerance = 0.00001
    pflag = False
    if 0:
        if np.abs(loss-lossref) > tolerance:
            print('lossref',lossref)
            print('loss',loss)
            pflag = True
        if np.abs(gradw[0][0]-gradref0) > tolerance:
            print('gradref0',gradref0)
            print('gradw0',gradw[0][0])
            pflag = True
        if np.abs(gradw[0][1]-gradref1) > tolerance:
            print('gradref1',gradref1)
            print('gradw1',gradw[0][1])
            pflag = True
        if np.abs(gradb[0]-gradrefb) > tolerance:
            print('gradrefb',gradrefb)
            print('gradb',gradb[0])
            pflag = True
    if pflag:
        print("epoch: {} \t x: {} \t xr:  \t yt: {}".format(it, x_train[i], y_train[i])) #, x_train[i].reshape((2,1)), y_train[i])) 
        print("epoch: {} \t w: {} \t b: {} \t y: {} \t gradw: {} \t gradb: {} \t loss: {}".format(it, w, b, y, gradw, gradb, loss))
        print('')

    _ = sess.run([train_op], feed_dict={x_input: x_train[i].reshape((1,2)), y_input: y_train[i]})

    y_preds.clear()
    for j in range(len(x_train)):
        y_pred = sess.run([y_value], feed_dict={x_input: x_train[j].reshape((1,2))}) 
        y_preds.append(y_pred)
    y_predsarray=np.array(y_preds)
    #print(y_train.shape)
    #print(y_predsarray.shape)
    lossall = ((y_train-y_predsarray)**2).mean(axis=None)
    #print(lossall)
    lossalls.append(lossall)
    lossis.append(loss)
    #print('')
sess.close()

start = 100
plt.plot(np.arange(start, len(lossalls), 1),lossalls[start:],color='r',alpha=0.5)
#plt.show()
plt.plot(np.arange(start, len(lossis), 1),lossis[start:],color='b',alpha=0.5)
Z0loss = ((Z)**2).mean(axis=None)
plt.axhline(y=Z0loss, linewidth=0.5, color='k')
print(Z0loss)
print(lossalls[-20:])

ax  = Axes3D(plt.figure())
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 
y = y_predsarray.reshape(Z.shape)
#print(y.shape)
#print(y)
#print(Z.shape)
#print(Z)
#axp = Axes3D(plt.figure())
ax.plot_surface(X, Y, y, rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 
plt.show()






exit()

ax  = Axes3D(plt.figure())
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 
y = y.reshape(Z.shape)
#axp = Axes3D(plt.figure())
ax.plot_surface(X, Y, y, rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 
plt.show()



exit()



sess = tf.Session()
x_input = tf.placeholder(tf.float32, shape=x_train.shape, name='x_input') 
y_input = tf.placeholder(tf.float32, shape=y_train.shape, name='y_input') 


y_value = tf.layers.dense(inputs=x_input, units=512, activation=tf.nn.relu, 
        kernel_initializer=tf.constant_initializer(2.0), bias_initializer=tf.constant_initializer(1.0))
y_value = tf.layers.dense(inputs=y_value, units=32, activation=tf.nn.relu, 
        kernel_initializer=tf.constant_initializer(2.0), bias_initializer=tf.constant_initializer(1.0))
y_value = tf.layers.dense(inputs=y_value, units=1, activation=tf.nn.softmax, 
        kernel_initializer=tf.constant_initializer(2.0), bias_initializer=tf.constant_initializer(1.0))

print(len(tf.trainable_variables()))
w_value = [v for v in tf.trainable_variables()][0]
b_value = [v for v in tf.trainable_variables()][1]



loss_op = tf.reduce_mean(tf.pow(y_input - y_value, 2))
train_op= tf.train.GradientDescentOptimizer(10.0).minimize(loss_op) 
grads_op= tf.gradients(loss_op, w_value) 

init = tf.global_variables_initializer() 
sess.run(init)

y_preds = []
for i in range(10): 
    _, w, b, y, grads, loss = sess.run([train_op, w_value, b_value, y_value, grads_op, loss_op], 
                                            feed_dict={x_input: x_train, y_input: y_train})
    #print("ep:{} x:{} w: {} b: {} y: {} y_t:{} loss:{:.2f} grads:{}".format(i, x_train[i], w, b, y, y_train[i], loss, gradients)) 
    print("ep:{} x: y:  y_t: loss:{} grads:{}".format(i, loss, grads)) 

    #y_preds.clear()
    #for j in range(len(x_train)):
    #    y_pred = sess.run([y_value], feed_dict={x_input: [[x_train[j]]]}) 
    #    y_preds.append(y_pred[0][0][0])
    #y_preds=np.array(y_preds)
    print(((y_train-y)**2).mean(axis=None))

    #if i%20==0:
ax  = Axes3D(plt.figure())
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 
y = y.reshape(Z.shape)
#axp = Axes3D(plt.figure())
ax.plot_surface(X, Y, y, rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 
plt.show()

sess.close()





exit()
plt.plot(X,y_train,color='r',alpha=0.5)
plt.plot(X,y,color='b',alpha=0.5)


figw = plt.figure(1)
axw = Axes3D(figw)
Xw = np.arange(-10, 10, 0.5) 
Yw = np.arange(-10, 10, 0.5) 
Xw, Yw = np.meshgrid(Xw, Yw) 

Zws = []
Tws = np.arange(-1, 1, 0.125) 

for Tw in Tws:
    Zw = Xw.copy()
    for i in range(len(Xw)):
        for j in range(len(Xw[i])):
            Zp = np.maximum(Xw[i][j]*X+Tw,0)
            Zp = np.maximum(Yw[i][j]*Zp,0)
            Zw[i][j] = ((Zp-Z)** 2).mean(axis=None)
    Zws.append(Zw)
Zws=np.array(Zws)

print(Xw.shape)
print(Yw.shape)
print(Zws.shape)
print(Xw[:3])
print(Yw[:3])
print(Zws.min())#[:3])
index = np.argmin(Zws)
index0= int(index/len(Xw.flatten()))
index1= index%len(Xw.flatten())
print(index)
print(index0)
print(index1)
print(Tws[index0])
print(Xw.flatten()[index1])
print(Yw.flatten()[index1])
print(Zws[index0].flatten()[index1])


axw.plot_surface(Xw, Yw, Zws[index0], rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 



plt.figure(2)
plt.plot(X,Z,color='r',alpha=0.5)
Zp = np.maximum(Xw.flatten()[index1]*X+Tws[index0],0)
Zp = np.maximum(Yw.flatten()[index1]*Zp,0)
plt.plot(X,Zp,color='b',alpha=0.5)

plt.show()



exit()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf 
from random import shuffle

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-5, 5, 0.25) 
Y = np.arange(-5, 5, 0.25) 
X, Y = np.meshgrid(X, Y) 
R = np.sqrt(X**2 + Y**2) 
#Z = X+Y+1.0
Z = R/10 + np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 

print(X.shape)
print(Y.shape)
print(Z.shape)
print(X)
print(Y)
print(Z)

print(((Z)**2).mean(axis=None))

figw = plt.figure()
axw = Axes3D(figw)
Xw = np.arange(-10, 10, 0.25) 
Yw = np.arange(-10, 10, 0.25) 
Xw, Yw = np.meshgrid(Xw, Yw) 
Zw = Xw.copy()
for i in range(len(Xw)):
    for j in range(len(Xw[i])):
        Zp = Xw[i][j]*X+Yw[i][j]*Y
        Zw[i][j] = ((Zp-Z)** 2).mean(axis=None)
axw.plot_surface(Xw, Yw, Zw, rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 

print(Xw.shape)
print(Yw.shape)
print(Zw.shape)
print(Xw[:3])
print(Yw[:3])
print(Zw.min())#[:3])
index = np.argmin(Zw)
print(Xw.flatten()[index])
print(Yw.flatten()[index])
print(Zw.flatten()[index])

#plt.show()


exit()
data_train = []
for i in range(len(Z)):
    for j in range(len(Z[i])):
        data_train.append((X[i][j],Y[i][j],Z[i][j]))
shuffle(data_train)
x_train = np.array([[data[0],data[1]] for data in data_train])
y_train = np.array([ [data[2]] for data in data_train])
#x0 = [xpair[0] for xpair in x_train]
#x1 = [xpair[1] for xpair in x_train]
#ax.scatter(x0, x1, y_train, c='r') 
print(x_train[0].shape)
print(y_train[0].shape)

sess = tf.Session()
x_input = tf.placeholder(tf.float32, shape=(2,2), name='x_input') 
y_input = tf.placeholder(tf.float32, name='y_input') 
#w_value = tf.Variable( tf.ones((2,)), name="weights")
#b_value = tf.Variable(tf.zeros((2,)), name="biases")
#w_value = tf.Variable(2.0, name='weight') 
#b_value = tf.Variable(1.0, name='biases') 

#w_value = tf.get_variable('w_value', shape=(2,), initializer=tf.constant_initializer(2.0))
#b_value = tf.get_variable('b_value', shape=(2,), initializer=tf.constant_initializer(1.0))
#y_value = tf.add(tf.multiply(x_input, w_value), b_value) 

y_value = tf.layers.dense(inputs=x_input, units=1, activation=None,#tf.nn.relu, 
        kernel_initializer=tf.constant_initializer(2.0), bias_initializer=tf.constant_initializer(1.0))
#w_value = tf.get_default_graph().get_tensor_by_name("weight:0") 
#b_value = tf.get_default_graph().get_tensor_by_name("bias:0") 
w_value = [v for v in tf.trainable_variables()][0]
b_value = [v for v in tf.trainable_variables()][1]



loss_op = tf.reduce_mean(tf.pow(y_input - y_value, 2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss_op) 
gradients_node = tf.gradients(loss_op, w_value) 

init = tf.global_variables_initializer() 
sess.run(init) 

for i in range(5): 
    _, w, b, y, gradients, loss = sess.run([train_op, w_value, b_value, y_value, gradients_node, loss_op], 
                                            feed_dict={x_input: [x_train[i],x_train[i+1]], y_input: [y_train[i],y_train[i+1]]}) 
    print("ep:{} x:{} w: {} b: {} y: {} y_t:{} loss:{:.2f} grads:{}".format(i, x_train[i], w, b, y, y_train[i], loss, gradients)) 

sess.close()



#plt.show()


exit()



sess = tf.Session()
x_input = tf.placeholder(tf.float32, shape=(2,), name='x_input') 
y_input = tf.placeholder(tf.float32, name='y_input') 
w = tf.Variable(2.0, name='weight') 
b = tf.Variable(1.0, name='biases') 
y = tf.add(tf.multiply(x_input, w), b) 
loss_op = tf.reduce_sum(tf.pow(y_input - y, 2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss_op) 

gradients_node = tf.gradients(loss_op, w) 
# print(gradients_node) 
# tf.summary.scalar('norm_grads', gradients_node) 
# tf.summary.histogram('norm_grads', gradients_node) 
# merged = tf.summary.merge_all() 
# writer = tf.summary.FileWriter('log') 

init = tf.global_variables_initializer() 
sess.run(init) 

x_pure = ( np.random.randint(-10, 10, 200), np.random.randint(-10, 10, 200) )
x_train = [(x_pure[0][i],x_pure[1][i]) for i in range(len(x_pure[0]))]#x_pure + ( np.random.randn(32) / 10, np.random.randn(32) / 10 )
y_train = [np.sqrt(xpair[0]**2 + xpair[1]**2) for xpair in x_train] #3 * x_pure + 2 + np.random.randn(32) / 10
y_train = [np.sin(y)+y/10 for y in y_train]



for i in range(20): 
    _, wv, bv, yv, gradients, loss = sess.run([train_op, w, b, y, gradients_node, loss_op], feed_dict={x_input: x_train[i], y_input: y_train[i]}) 
    print("epoch: {} \t w: {} \t b: {} \t y: {} \t gradients: {} \t loss: {}".format(i, wv, bv, yv, gradients, loss)) 

sess.close()


exit()

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-10, 10, 0.25) 
Y = np.arange(-10, 10, 0.25) 
X, Y = np.meshgrid(X, Y) 
R = np.sqrt(X**2 + Y**2) 
Z = np.sin(R)+R/10
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.2) 

ax.scatter(x_pure[0], x_pure[1], y_train, c='r') 
plt.show()




exit()

for i in range(20): 
    _, wv, bv, yv, gradients, loss = sess.run([train_op, w, b, y, gradients_node, loss_op], feed_dict={x_input: x_train[i], y_input: y_train[i]}) 
    print("epoch: {} \t w: {} \t b: {} \t y: {} \t gradients: {} \t loss: {}".format(i, wv, bv, yv, gradients, loss)) 

sess.close()


exit()



exit()

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-10, 10, 0.25) 
Y = np.arange(-10, 10, 0.25) 
X, Y = np.meshgrid(X, Y) 
R = np.sqrt(X**2 + Y**2) 
Z = np.sin(R)+R/10
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow') 
plt.show()














exit()


data = np.random.randint(0, 255, size=[40, 40, 40])
x, y, z = data[0], data[1], data[2] 
ax = plt.subplot(111, projection='3d') 
ax.scatter(x[:10], y[:10], z[:10], c='y') 
ax.scatter(x[10:20], y[10:20], z[10:20], c='r') 
ax.scatter(x[30:40], y[30:40], z[30:40], c='g') 
ax.set_zlabel('Z')
ax.set_ylabel('Y') 
ax.set_xlabel('X')
plt.show()





exit()







inputdata = np.array(10,10)

for i in range(len(inputdata)):
    for j in range(len(inputdata[i])):
            inputdata[i][j] = sin(i)+sin(j)











import numpy, os, argparse, json, easydict
import matplotlib.pyplot as plt
from itertools import cycle
COLORS = cycle(['black', 'red', 'orange', 'green', 'cyan', 'blue', 'purple'])
def plot(fdata,color):
    lines=fdata.read().splitlines()
    for i in range(0,len(lines)):
        datas = lines[i].split(",")
        datas = datas[:len(datas)-1]
        x = [int(data.split("|")[0]) for data in datas]
        y = [float(data.split("|")[1]) for data in datas]
        avgnum = 10
        xmean = [x[i] for i in range(len(x)-avgnum+1)]
        ymean = [numpy.mean(y[i:i+avgnum]) for i in range(len(y)-avgnum+1)]
        plt.plot(x,y,color=color,alpha=0.1)
        plt.plot(xmean,ymean,color=color,alpha=1.0)

parser = argparse.ArgumentParser(description="Atari A2C TensorFlow experiments")
parser.add_argument('--config', default=None, type=str, help='Configuration file')
parser.add_argument('--seed', default=666, type=str, help='Env seed')
args = parser.parse_args()
with open(args.config, 'r') as config_file:
    config_args_dict = json.load(config_file)
config_args = easydict.EasyDict(config_args_dict)
with open('./myenv/envinfo.json', 'w') as fenvinfo:
    print(json.dumps(config_args),file=fenvinfo)
config_args.env_seed       = int(args.seed)
config_args.experiment_dir = args.config.replace('.', '_') + str(config_args.env_seed) + "/"
config_args.checkpoint_dir = config_args.experiment_dir + 'checkpoints/'
config_args.summary_dir    = config_args.experiment_dir + 'summaries/'
config_args.output_dir     = config_args.experiment_dir + 'output/'
config_args.test_dir       = config_args.experiment_dir + 'test/'
dirs = [config_args.checkpoint_dir, config_args.summary_dir, config_args.output_dir, config_args.test_dir]
for dir_ in dirs:
    if not os.path.exists(dir_): os.makedirs(dir_)

plt.rcParams['figure.figsize'] = (24.0, 12.0)#(8.0, 4.0)
for i in range(config_args.num_envs):#write data
    fdata = open(config_args.summary_dir+'data'+str(i),'r')
    color=next(COLORS)
    plot(fdata,color)
    fdata.close()
axes = plt.gca()
totalsteps = int(config_args.train_iterations)*int(config_args.unroll_time_steps)
axes.set_xticks(numpy.arange(0,totalsteps,totalsteps/10))
axes.set_yticks(numpy.arange(0,300,25))
plt.xlim([0,totalsteps])
plt.ylim([0,300])
plt.savefig(config_args.experiment_dir+"datas.png", dpi=100)
plt.savefig(args.config.replace('.', '_') + str(config_args.env_seed) + ".png", dpi=100)
#plt.show()
