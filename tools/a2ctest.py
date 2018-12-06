import tensorflow as tf
import numpy as np
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
def noise_and_argmax(logits): # Add noise then take the argmax
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)
#from model_layers import conv2d, flatten, dense
import sys, gym, tqdm, argparse, json, easydict, os, time, random, pprint, myenv
import matplotlib.pyplot as plt
from itertools import cycle
###########################################################################
class BaseModel:
    def __init__(self,env,args):
        self.batch_num, self.stack_num, self.input_shape, self.output_num = args.batch_num, args.stack_num, env.observation_space.shape, env.action_space.n
        self.unitslist = [int(num) for num in args.hypers.split('_')]

        self.input = tf.placeholder(tf.float32, [self.batch_num]+list(self.input_shape), name='input')
        initializer= tf.truncated_normal_initializer
        self.h_values = []
        h_value = self.input
        self.h_values.append(h_value)
        for i,units in enumerate(self.unitslist):
            h_value = tf.layers.dense(inputs=h_value, units=units, activation=None, kernel_initializer=initializer, use_bias=False)
            h_value = tf.nn.relu(h_value)
            self.h_values.append(h_value)

        self.ay_value = tf.layers.dense(inputs=h_value, units=self.output_num, activation=None, kernel_initializer=initializer, use_bias=False)
        self.h_values.append(self.ay_value)
        self.policy_logits    = tf.nn.softmax(self.ay_value)
        self.target_actor     = tf.placeholder(tf.float32, [self.batch_num]+list([self.output_num]), name='target_actor')
        self.loss_actor       = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.ay_value, labels=self.target_actor))

        self.cy_value = tf.layers.dense(inputs=h_value, units=1, activation=None, kernel_initializer=initializer, use_bias=False)
        self.h_values.append(self.cy_value)
        self.value_function   = self.cy_value
        self.target_critic    = tf.placeholder(tf.float32, [self.batch_num]+list([1]), name='target_critic')
        self.loss_critic      = tf.reduce_mean(tf.square(self.target_critic - self.value_function))
class BaseFitter:
    def __init__(self,env,sess,args):
        self.sess, self.model = sess, BaseModel(env,args)
        self.LR_A, self.LR_C, self.GAMMA, self.penalize = args.actor_learning_rate, args.critic_learning_rate, args.reward_discount_factor, args.penalize
        #self.actlr, self.crtlr = tf.placeholder(tf.float32, []), tf.placeholder(tf.float32, [])
        self.train_actor_ops = tf.train.AdamOptimizer(0.001).minimize(self.model.loss_actor)
        self.train_critic_ops= tf.train.AdamOptimizer(0.005).minimize(self.model.loss_critic)
        self.loss = self.model.loss_actor + self.model.loss_critic*5.5 #- self.entropy*0.01
        self.optimize = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
    def getaction(self, s):
        return np.random.choice(list(range(self.model.output_num)), 1, p=self.sess.run(self.model.policy_logits, feed_dict={self.model.input: s}).ravel())[0]
    def update(self, s_old, a, r, s_new, failed):
        target_critic = np.zeros((self.model.batch_num, 1))
        target_actor  = np.zeros((self.model.batch_num, self.model.output_num))
        V_s_old = self.sess.run(self.model.value_function, feed_dict={self.model.input: s_old})
        V_s_new = self.sess.run(self.model.value_function, feed_dict={self.model.input: s_new})
        if failed: V_s_new = self.penalize - 1.0 ###
        target_critic[0][0]= r + self.GAMMA * V_s_new
        target_actor[0][a] = r + self.GAMMA * V_s_new - V_s_old
        self.sess.run([self.optimize],
        #self.sess.run([self.train_critic_ops, self.train_actor_ops],
            feed_dict={self.model.input: s_old, self.model.target_critic: target_critic, self.model.target_actor: target_actor})
class BaseTrainer:
    def __init__(self,env,sess,args):
        self.env, self.fitter = env, BaseFitter(env,sess,args)
        self.target_score, self.avg_num, self.solved_rate = args.target_score, args.avg_num, args.solved_rate
        self.max_steps, self.max_episodes = args.max_steps, args.max_episodes
    def checksolved(self, episode_rewards):
        return np.mean(episode_rewards[-min(self.avg_num,len(episode_rewards)):]) > self.target_score*self.solved_rate
    def checkfailed(self, episode_rewards):
        return episode_rewards[-1] < self.target_score
    def s_reshaper(self, s):
        return np.reshape(s, [self.fitter.model.batch_num]+list(s.shape))
    def fit(self,f):
        episode_solved, episode_rewards = [False], [0.0]
        obs = self.env.reset()
        obs = self.s_reshaper(obs)
        for t in tqdm.tqdm(range(self.max_steps)):
            action = self.fitter.getaction(obs)
            new_obs, rew, done, info = self.env.step(action)### batched action
            new_obs = self.s_reshaper(new_obs)
            episode_rewards[-1]+=rew
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
        trainer = BaseTrainer(env,sess,config_args)
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
