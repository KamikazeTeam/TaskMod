import tensorflow as tf
import numpy as np
###########################################################################
def variable_with_weight_decay(kernel_shape, initializer, wd):
    #Create a variable with L2 Regularization (Weight Decay)
    #:param kernel_shape: the size of the convolving weight kernel.
    #:param wd:(weight decay) L2 regularization parameter.
    #:return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)
    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    return w
def get_deconv_filter(f_shape, l2_strength):
    #The initializer for the bilinear convolution transpose filters
    #:param f_shape: The shape of the filter used in convolution transpose.
    #:param l2_strength: L2 regularization parameter.
    #:return weights: The initialized weights.
    width = f_shape[0]
    height = f_shape[0]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return variable_with_weight_decay(weights.shape, init, l2_strength)
def noise_and_argmax(logits): # Add noise then take the argmax
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)
def openai_entropy(logits): # Entropy proposed by OpenAI in their A2C baseline
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)
def softmax_entropy(p0): # Normal information theory entropy by Shannon
    return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis=1)
def mse(predicted, ground_truth): # Mean-squared error
    return tf.square(predicted - ground_truth) / 2.
def orthogonal_initializer(scale=1.0):#The unused variables are just for passing in tensorflow
    def _ortho_init(shape, dtype, partition_info=None): # Orthogonal Initializer that uses SVD.
        shape = tuple(shape)
        if len(shape) == 2: flat_shape = shape
        elif len(shape) == 4: flat_shape = (np.prod(shape[:-1]), shape[-1])# assumes NHWC
        else: raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init
from model_layers import mse, openai_entropy
from model_layers import conv2d, flatten, dense
from model_layers import orthogonal_initializer, noise_and_argmax
###########################################################################
import sys, gym, tqdm, argparse, json, easydict, os, time, random, pprint, myenv
import matplotlib.pyplot as plt
from itertools import cycle
class DenseModel:
    def __init__(self,env,args):# have to be linear to make sure the convergence of actor. But linear approximator seems hardly learns the correct Q.
        self.batch_num, self.stack_num, self.input_shape, self.output_num = args.batch_num, args.stack_num, env.observation_space.shape, env.action_space.n
        self.unitslist = [int(num) for num in args.hypers.split('_')]

        self.input_actor  = tf.placeholder(tf.float32, [self.batch_num]+list(self.input_shape), name='input_actor')
        self.target_actor = tf.placeholder(tf.float32, [self.batch_num]+list([self.output_num]), name='target_actor')
        ainitializer = tf.truncated_normal_initializer#tf.truncated_normal()#tf.random_uniform() #tf.constant_initializer(1.0)
        self.ah_values = []
        ah_value = self.input_actor
        self.ah_values.append(ah_value)
        for i,units in enumerate(self.unitslist):
            #if i==0:
            #    ah_value = tf.layers.dense(inputs=ah_value, units=units, activation=None, kernel_initializer=ainitializer, use_bias=False)
            #    #ah_value = tf.layers.batch_normalization(inputs=ah_value,scale=False,training=True)
            #    ah_value = tf.nn.relu(ah_value)
            #h_valueo= tf.identity(h_value)
            #ah_value = tf.layers.batch_normalization(inputs=ah_value,scale=False,training=True)#tf.contrib.layers.batch_norm(h_value)
            #ah_value = tf.nn.relu(ah_value)
            ah_value = tf.layers.dense(inputs=ah_value, units=units, activation=None, kernel_initializer=ainitializer, use_bias=False)
            #ah_value = tf.layers.batch_normalization(inputs=ah_value,scale=False,training=True)
            ah_value = tf.nn.relu(ah_value)
            #ah_value = tf.layers.dense(inputs=ah_value, units=units, activation=None, kernel_initializer=ainitializer, use_bias=False)
            #h_value = h_value + h_valueo
            self.ah_values.append(ah_value)
        self.ay_value = tf.layers.dense(inputs=ah_value, units=self.output_num, activation=None, kernel_initializer=ainitializer, use_bias=False)#tf.nn.softmax
        self.ah_values.append(self.ay_value)
        self.policy_logits    = tf.nn.softmax(self.ay_value)
        self.loss_actor       = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.ay_value, labels=self.target_actor))

        self.input_critic = tf.placeholder(tf.float32, [self.batch_num]+list(self.input_shape), name='input_critic')
        self.target_critic= tf.placeholder(tf.float32, [self.batch_num]+list([1]), name='target_critic')
        cinitializer = tf.truncated_normal_initializer#tf.random_uniform() #tf.constant_initializer(1.0)
        self.ch_values = []
        ch_value = self.input_critic
        self.ch_values.append(ch_value)
        for i,units in enumerate(self.unitslist):
            #if i==0:
            #    ch_value = tf.layers.dense(inputs=ch_value, units=units, activation=None, kernel_initializer=cinitializer, use_bias=False)
            #    #ch_value = tf.layers.batch_normalization(inputs=ch_value,scale=False,training=True)
            #    ch_value = tf.nn.relu(ch_value)
            #h_valueo= tf.identity(h_value)
            #ch_value = tf.layers.batch_normalization(inputs=ch_value,scale=False,training=True)#tf.contrib.layers.batch_norm(h_value)
            #ch_value = tf.nn.relu(ch_value)
            ch_value = tf.layers.dense(inputs=ch_value, units=units, activation=None, kernel_initializer=cinitializer, use_bias=False)
            #ch_value = tf.layers.batch_normalization(inputs=ch_value,scale=False,training=True)
            ch_value = tf.nn.relu(ch_value)
            #ch_value = tf.layers.dense(inputs=ch_value, units=units, activation=None, kernel_initializer=cinitializer, use_bias=False)
            #h_value = h_value + h_valueo
            self.ch_values.append(ch_value)
        self.cy_value = tf.layers.dense(inputs=ch_value, units=1, activation=None, kernel_initializer=cinitializer, use_bias=False)#tf.nn.softmax
        self.ch_values.append(self.cy_value)
        self.value_function   = self.cy_value
        self.loss_critic      = tf.reduce_sum(tf.square(self.target_critic - self.value_function))
class CNNFitter:
    def __init__(self,sess,model,args):
        self.sess, self.model = sess, model
        self.GAMMA = args.reward_discount_factor
        self.actions        = tf.placeholder(tf.int32, [None])
        self.rewards        = tf.placeholder(tf.float32, [None])
        self.advantages     = tf.placeholder(tf.float32, [None])
        negative_log_prob_action  = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model.policy_logits, labels=self.actions)
        self.policy_gradient_loss = tf.reduce_mean(self.advantages * negative_log_prob_action)
        self.value_function_loss  = tf.reduce_mean(mse(tf.squeeze(self.model.value_function), self.rewards))
        self.entropy              = tf.reduce_mean(openai_entropy(self.model.policy_logits))
        self.loss = self.policy_gradient_loss - self.entropy*0.01 + self.value_function_loss*0.5
        self.optimize = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
    def getaction(self, s):
        #return noise_and_argmax(self.sess.run(self.model.policy_logits, feed_dict={self.model.input_actor: s}))
        return np.random.choice(list(range(self.model.output_num)), 1, p=self.sess.run(self.model.policy_logits, feed_dict={self.model.input_actor: s}).ravel())[0]
    def update(self, s_old, a, r, s_new, done):
        values  = self.sess.run(self.model.value_function, feed_dict={self.model.input_actor: s_old})[:, 0]
        rewards = 
        advantages = rewards - values
        self.sess.run([self.optimize], {self.model.input_actor: s_old, self.actions: a, self.rewards: rewards, self.advantages: advantages})
###########################################################################
class Fitter:
    def __init__(self,sess,model,args):
        self.sess, self.model = sess, model
        self.LR_A, self.LR_C, self.GAMMA = args.actor_learning_rate, args.critic_learning_rate, args.reward_discount_factor
        self.actlr = tf.placeholder(tf.float32, [])
        self.crtlr = tf.placeholder(tf.float32, [])
        self.train_actor_ops = tf.train.AdamOptimizer(self.actlr).minimize(self.model.loss_actor)
        self.train_critic_ops= tf.train.AdamOptimizer(self.crtlr).minimize(self.model.loss_critic)
        self.sess.run(tf.global_variables_initializer())
    def getaction(self, s):
        return np.random.choice(list(range(self.model.output_num)), 1, p=self.sess.run(self.model.policy_logits, feed_dict={self.model.input_actor: s}).ravel())[0]
    def update(self, s_old, a, r, s_new, done):
        value_actlr, value_crtlr = self.LR_A, self.LR_C#*(1-t/MAX_STEPS), LR_C*(1-t/MAX_STEPS)
        target_critic = np.zeros((self.model.batch_num, 1))
        target_actor  = np.zeros((self.model.batch_num, self.model.output_num))
        V_s_old = self.sess.run(self.model.value_function, feed_dict={self.model.input_critic: s_old})
        V_s_new = self.sess.run(self.model.value_function, feed_dict={self.model.input_critic: s_new})
        if done: V_s_new = 0.0 # The value function of s_new must be zero because the state leads to game end
        #if do not do this, experiments show that at last there will be a good probability that every episode end at only 9-10 steps
        target_critic[0][0]= r + self.GAMMA * V_s_new
        target_actor[0][a] = r + self.GAMMA * V_s_new - V_s_old
        self.sess.run([self.train_critic_ops, self.train_actor_ops], feed_dict={ self.actlr: value_actlr, self.crtlr: value_crtlr,
            self.model.input_critic: s_old, self.model.target_critic: target_critic, self.model.input_actor: s_old,  self.model.target_actor: target_actor})
###########################################################################
class Trainer:
    def __init__(self,env,model,fitter,args):
        self.env, self.model, self.fitter, self.experiment_dir = env, model, fitter, args.experiment_dir
        self.target_score, self.avg_num, self.solved_rate, self.penalize = args.target_score, args.avg_num, args.solved_rate, args.penalize
        self.max_steps, self.max_episodes = args.max_steps, args.max_episodes
    def checksolved(self, episode_rewards):
        return np.mean(episode_rewards[-min(self.avg_num,len(episode_rewards)):]) > self.target_score*self.solved_rate
    def checkfailed(self, episode_rewards):
        return episode_rewards[-1] < self.target_score
    def s_reshaper(self, s):
        return np.reshape(s, [self.model.batch_num]+list(s.shape))
    def fit(self,f):
        episode_solved, episode_rewards = [False], [0.0]
        obs = self.env.reset()
        obs = self.s_reshaper(obs)
        for t in tqdm.tqdm(range(self.max_steps)):
            action = self.fitter.getaction(obs)
            new_obs, rew, done, info = self.env.step(action)### batched action
            new_obs = self.s_reshaper(new_obs)
            episode_rewards[-1]+=rew
            if done and self.checkfailed(episode_rewards): rew = self.penalize
            if not episode_solved[-1]: self.fitter.update(obs, action, rew, new_obs, done and self.checkfailed(episode_rewards))
            obs = new_obs
            if done:
                print(int(episode_rewards[-1]),end='|',file=f,flush=True)
                episode_solved.append(self.checksolved(episode_rewards))
                if len(episode_rewards) >= self.max_episodes: break
                episode_rewards.append(0.0)
                obs = self.env.reset()
                obs = self.s_reshaper(obs)
        return episode_rewards, episode_solved
###########################################################################
def setsession(randseed):
    tfconfig = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=tfconfig)
    random.seed(randseed)
    np.random.seed(randseed)
    tf.set_random_seed(randseed)
    return sess
def main():
    parser = argparse.ArgumentParser(description="a2ctest")
    parser.add_argument('--config', default=None, type=str, help='Configuration file')
    parser.add_argument('--seed', default=666, type=str, help='Env seed')
    parser.add_argument('--finalseed', default=666, type=str, help='Env finalseed')
    parser.add_argument('--to_train', default=False, type=str, help='To train')
    parser.add_argument('--hypers', default=None, type=str, help='Hyperparameter')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config_args_dict = json.load(config_file)
    config_args = easydict.EasyDict(config_args_dict)
    with open('./myenv/envinfo.json', 'w') as fenvinfo:
        print(json.dumps(config_args),file=fenvinfo)
    config_args.env_seed = int(args.seed)
    config_args.to_train = bool(args.to_train)
    config_args.hypers   = args.hypers
    config_args.experiment_dir = "./"+args.config.replace('.', '_')+'_'+args.hypers+'_'# + str(config_args.env_seed) + "/"
    config_args.checkpoint_dir = config_args.experiment_dir + 'checkpoints/'
    config_args.summary_dir    = config_args.experiment_dir + 'summaries/'
    config_args.output_dir     = config_args.experiment_dir + 'output/'
    config_args.test_dir       = config_args.experiment_dir + 'test/'
    dirs = [config_args.checkpoint_dir, config_args.summary_dir, config_args.output_dir, config_args.test_dir]
    #for dir_ in dirs:
    #    if not os.path.exists(dir_): os.makedirs(dir_)
    config_args.curvename, config_args.doneepsname = config_args.experiment_dir+"curve", config_args.experiment_dir+"doneeps"
    flog = open(config_args.experiment_dir+'config_args_log','a')
    starttime = time.time()

    sess = setsession(config_args.env_seed)
    env  = gym.make(config_args.env_name)
    env.seed(config_args.env_seed)
    if config_args.policy_name=='CNNPolicy':
        model = CNNModel(env,config_args)
        fitter = CNNFitter(sess,model,config_args)
        trainer = CNNTrainer(env,model,fitter,config_args)
    else:
        model = DenseModel(env,config_args)
        fitter = Fitter(sess,model,config_args)
        trainer = Trainer(env,model,fitter,config_args)
    print(args.seed,':',args.finalseed)
    fcurve = open(config_args.curvename,'a')
    print('',file=fcurve)
    episode_rewards, episode_solved = trainer.fit(fcurve)
    fcurve.close()

    fdoneeps = open(config_args.doneepsname,'a')
    episode_solved.reverse()
    print(episode_solved.index(False),end=',',file=fdoneeps)
    fdoneeps.close()
    pprint.pprint(config_args,flog)
    endtime = time.time()
    print(time.ctime(starttime),file=flog)
    print(time.ctime(endtime),file=flog)
    print((endtime-starttime)/60,'minutes',file=flog)
    print((endtime-starttime)/3600,'hours',file=flog)
    flog.close()
    if args.seed==args.finalseed or int(args.seed)%10==0:
        doneeps  = [int(doneep) for doneep in open(config_args.doneepsname,'r').read().splitlines()[0].split(",")[:-1]]
        plt.figure(111)
        bins = np.linspace(0,config_args.max_episodes,50)
        plt.hist(doneeps, bins=bins, normed=False,facecolor='r',edgecolor='r',hold=0,alpha=0.2)
        plt.savefig(config_args.doneepsname+'.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
        COLORS = cycle(['black', 'red', 'orange', 'green', 'cyan', 'blue', 'purple'])
        lines=open(config_args.curvename,"r").read().splitlines()
        for i in range(0,len(lines)):
            color=next(COLORS)
            record, avgnum = [float(strs) for strs in lines[i].split("|")[:-1]], 50 ###
            recordmean = [np.mean(record[j:j+avgnum]) for j in range(len(record)-avgnum+1)]
            plt.plot(record,label=i,color=color,alpha=0.1)
            plt.plot(recordmean,label=i,color=color,alpha=1.0)
        axes = plt.gca()
        plt.savefig(config_args.curvename+'.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
if __name__ == '__main__':
    main()

"""class CNNModel:
    def __init__(self, args, X_input, num_actions, reuse=False, is_training=False):
        self.P_input = X_input
        self.cast = tf.transpose(self.P_input[:,:,:,:,0],[0,2,3,1])
        self.cast = tf.cast(self.cast, tf.float32) / 255.0
        self.convs, self.convouts = list(args.convs), []
        self.denss, self.densouts = list(args.denss), []
        with tf.variable_scope("policy", reuse=reuse):
            self.conv = self.cast
            for i,conv in enumerate(self.convs):
                convout = conv2d('conv'+str(i), self.conv, num_filters=conv[0], kernel_size=(conv[1], conv[2]), padding='VALID', stride=(conv[3], conv[4]),
                        initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu, is_training=is_training)
                self.conv = convout
                self.convouts.append(convout)
            self.conv_flattened = flatten(self.conv)
            self.dens = self.conv_flattened
            for i,dens in enumerate(self.denss):
                densout = dense('dens'+str(i), self.dens, output_dim=dens,
                        initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu, is_training=is_training)
                self.dens = densout
                self.densouts.append(densout)
            self.policy_logits  = dense('policy_logits',  self.dens, output_dim=num_actions, initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=is_training)
            self.value_function = dense('value_function', self.dens, output_dim=1          , initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=is_training)
            with tf.name_scope('action'):
                self.action_s = noise_and_argmax(self.policy_logits)
            with tf.name_scope('value'):
                self.value_s = self.value_function[:, 0]"""
