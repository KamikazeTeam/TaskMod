import tensorflow as tf
import numpy as np
import sys, gym, tqdm, argparse, json, easydict, os, time, random, pprint, myenv
import matplotlib.pyplot as plt
from itertools import cycle
class Dense:
    def __init__(self,env,args):
        self.batch_num, self.stack_num = None, int(args.hypers_model[0]) #args.batch_num, args.stack_num *env.env_num
        self.unitslist = [int(num) for num in args.hypers_model[1].split('_')]
        self.batch_normal = bool(int(args.hypers_model[2]))
        self.input_shape, self.output_num = [self.stack_num]+list(env.observation_space.shape), env.action_space.n
        self.input = tf.placeholder(tf.float32, [self.batch_num]+self.input_shape, name='input')
        self.train = tf.placeholder(tf.bool, name='phase')
        initializer= tf.truncated_normal_initializer
        self.shaped_input = tf.layers.flatten(self.input) ######
        self.h_values = []
        h_value = self.shaped_input
        self.h_values.append(h_value)
        for i,units in enumerate(self.unitslist):
            h_value = tf.layers.dense(inputs=h_value, units=units, activation=None, kernel_initializer=initializer, use_bias=False)
            if self.batch_normal:
                h_value = tf.layers.batch_normalization(inputs=h_value,scale=False,training=self.train)
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
        self.GAMMA, self.penalize = float(args.hypers_fitter[0]), float(args.hypers_fitter[1]) #args.reward_discount_factor, args.penalize
        self.lr, self.ratio = float(args.hypers_fitter[2]), float(args.hypers_fitter[3])
        self.sess, self.model = sess, Dense(env,args)
        self.loss = self.model.loss_actor + self.model.loss_critic*self.ratio #- self.entropy*0.01 ######
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
    def getaction(self, s):
        actions = []
        p = self.sess.run(self.model.policy_logits, feed_dict={self.model.input: s, self.model.train: False}) #p.ravel()
        for i in range(len(p)):
            action = np.random.choice(list(range(self.model.output_num)), 1, p=p[i])[0]
            actions.append(action)
        return actions
    def update(self, s_old, a, r, s_new, failed):
        batch_num = len(a)
        target_critic = np.zeros((batch_num, 1))
        target_actor  = np.zeros((batch_num, self.model.output_num))
        V_s_old = self.sess.run(self.model.value_function, feed_dict={self.model.input: s_old, self.model.train: False})
        V_s_new = self.sess.run(self.model.value_function, feed_dict={self.model.input: s_new, self.model.train: False})
        for i in range(batch_num):
            if failed[i]: V_s_new[i] = self.penalize - 1.0 ######
            target_critic[i][0] = r[i] + self.GAMMA * V_s_new[i]
            target_actor[i][a[i]] = r[i] + self.GAMMA * V_s_new[i] - V_s_old[i]
        self.sess.run([self.optimize], feed_dict={self.model.input: s_old,
            self.model.target_critic: target_critic, self.model.target_actor: target_actor, self.model.train: True})
class MultiEnv:
    def __init__(self, env_name, env_seed, env_num):
        self.envs = []
        for ienv in range(env_num):
            env = gym.make(env_name)
            env.seed(env_seed+ienv)
            self.envs.append(env)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    def reseti(self,i):
        obs = self.envs[i].reset()
        obs = np.multiply(obs,np.array([10,1,10,1]))
        return obs
    def step(self, actions):
        obss, rews, dones, infos = [], [], [], []
        for ienv in range(len(self.envs)):
            obs, rew, done, info = self.envs[ienv].step(actions[ienv])
            obs = np.multiply(obs,np.array([10,1,10,1]))
            obss.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
        return obss, rews, dones, infos#np.array(obss), np.array(rews), np.array(dones), np.array(infos)
class BaseTrainer:
    def __init__(self,sess,args):
        self.env_num = int(args.hypers_trainer[0])
        self.roll_num = int(args.hypers_trainer[1])
        self.batch_num = self.env_num * self.roll_num
        self.env = MultiEnv(args.env_name, args.env_seed, self.env_num)
        self.fitter = BaseFitter(self.env,sess,args)
        self.target_score, self.avg_num, self.solved_rate, self.max_unsolvedp = args.target_score, args.avg_num, args.solved_rate, float(args.hypers_trainer[2])#args.max_unsolvedp
        self.max_steps, self.max_episodes = args.max_steps, args.max_episodes
    def create_fs(self, fname, message=None):
        fs = []
        for i in range(self.env_num):
            f = open(fname+str(i),'a')
            if message!=None: print(message,file=f)
            fs.append(f)
        return fs
    def close_fs(self, fs):
        for i in range(self.env_num):
            fs[i].close()
    def create_ls(self,initvalue):
        ls = []
        for i in range(self.env_num):
            ls.append([initvalue])
        return ls
    def add_ls_last(self, ls, value):
        for i in range(self.env_num):
            ls[i][-1]+=value[i]
    def check_ls_last(self, ls, value):
        results = []
        for i in range(self.env_num):
            results.append(ls[i][-1]==value)
        return np.array(results)
    def checkfailed(self, done, episode_rewards):
        failed = []
        for i in range(self.env_num):
            failed.append( done[i] and episode_rewards[i][-1] < self.target_score)
        return failed
    def checksolved(self, episode_rewards):
        return np.mean(episode_rewards[-min(self.avg_num,len(episode_rewards)):]) > self.target_score*self.solved_rate
    def obs_stack_update(self, new_obs, old_obs_stack):
        updated_obs_stack = np.roll(old_obs_stack, shift=-1, axis=1)
        updated_obs_stack[:,-1,:] = new_obs[:]#,:]
        return updated_obs_stack
    def obs_stack_reseti(self, obs_stack, i):
        reseted_obs_stack = obs_stack
        reseted_obs_stack[i] *= 0
        reseted_obs_stack[i][-1] = self.env.reseti(i)
        return reseted_obs_stack
    def fit(self,config_args):
        fcurves = self.create_fs(config_args.curvesname,'')
        episode_solved, episode_rewards = self.create_ls(False), self.create_ls(0.0)
        obs_stack = np.zeros([self.env_num]+self.fitter.model.input_shape, dtype=np.float32)
        for i in range(self.env_num):
            obs_stack = self.obs_stack_reseti(obs_stack,i)
        for t in tqdm.tqdm(range(self.max_steps)):
            action = self.fitter.getaction(obs_stack)
            new_obs, rew, done, info = self.env.step(action)
            new_obs_stack = self.obs_stack_update(new_obs, obs_stack)
            self.add_ls_last(episode_rewards, rew)
            unsolved = np.mean(self.check_ls_last(episode_solved,False))
            if unsolved > self.max_unsolvedp:
                self.fitter.update(obs_stack, action, rew, new_obs_stack, self.checkfailed(done, episode_rewards))
            obs_stack = new_obs_stack
            for i in range(self.env_num):
                if done[i]:
                    print(int(episode_rewards[i][-1]),end='|',file=fcurves[i],flush=True)
                    episode_solved[i].append(self.checksolved(episode_rewards[i]))
                    episode_rewards[i].append(0.0)
                    obs_stack = self.obs_stack_reseti(obs_stack,i)
            if min([len(episode_rewards[i]) for i in range(self.env_num)]) > self.max_episodes: break
        self.close_fs(fcurves)
        fdoneeps = self.create_fs(config_args.doneepsname)
        for i in range(self.env_num):
            episode_solved[i] = episode_solved[i][:self.max_episodes]
            episode_solved[i].reverse()
            print(self.max_episodes-episode_solved[i].index(False),end=',',file=fdoneeps[i])
        self.close_fs(fdoneeps)
        return episode_rewards, episode_solved, self.env_num
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
    parser.add_argument('--hypers', default=None, type=str, help='Hyperparameter')
    parser.add_argument('--seed', default=666, type=str, help='Env seed')
    parser.add_argument('--finalseed', default=666, type=str, help='Env finalseed')
    parser.add_argument('--to_train', default=False, type=str, help='To train')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config_args_dict = json.load(config_file)
    config_args = easydict.EasyDict(config_args_dict)
    with open('./myenv/envinfo.json', 'w') as fenvinfo:
        print(json.dumps(config_args),file=fenvinfo)
    config_args.hypers   = args.hypers.split('+')
    config_args.hypers_trainer = config_args.hypers[0].split('|')
    config_args.hypers_fitter  = config_args.hypers[1].split('|')
    config_args.hypers_model   = config_args.hypers[2].split('|')
    config_args.env_seed = int(args.seed)
    config_args.to_train = bool(args.to_train)
    config_args.experiment_dir = args.config.replace('.', '_')+'_'+args.hypers+'_/'# + str(config_args.env_seed) + "/"
    config_args.checkpoint_dir = config_args.experiment_dir + 'checkpoints/'
    config_args.summary_dir    = config_args.experiment_dir + 'summaries/'
    config_args.output_dir     = config_args.experiment_dir + 'output/'
    config_args.test_dir       = config_args.experiment_dir + 'test/'
    dirs = [config_args.checkpoint_dir, config_args.summary_dir, config_args.output_dir, config_args.test_dir]
    for dir_ in dirs:
        if not os.path.exists(dir_): os.makedirs(dir_)
    config_args.configsname = config_args.experiment_dir+"configs"
    config_args.curvesname  = config_args.experiment_dir+"curves"
    config_args.doneepsname = config_args.experiment_dir+"doneeps"

    starttime = time.time()
    sess = setsession(config_args.env_seed)
    trainer = BaseTrainer(sess,config_args)
    print(args.seed,':',args.finalseed)
    episode_rewards, episode_solved, env_num = trainer.fit(config_args)
    endtime = time.time()

    fconfigs = open(config_args.configsname,'a')
    pprint.pprint(config_args,fconfigs)
    print(time.ctime(starttime),file=fconfigs)
    print(time.ctime(endtime),file=fconfigs)
    print((endtime-starttime)/60,'minutes',file=fconfigs)
    print((endtime-starttime)/3600,'hours',file=fconfigs)
    fconfigs.close()
    if args.seed==args.finalseed or int(args.seed)%10==0:
        doneeps = []
        for i in range(env_num):
            doneeps.append([int(doneep) for doneep in open(config_args.doneepsname+str(i),'r').read().splitlines()[-1].split(",")[:-1]])
        doneeps = np.array(doneeps).mean(axis=0)
        plt.figure(111)
        bins = np.linspace(0,config_args.max_episodes,50)
        plt.hist(doneeps, bins=bins, normed=False,facecolor='r',edgecolor='r',hold=0,alpha=0.2)
        plt.savefig(config_args.doneepsname.replace('/','T')+'.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
        COLORS = cycle(['black', 'red', 'orange', 'green', 'cyan', 'blue', 'purple'])
        lines = []
        for i in range(env_num):
            lines.append(open(config_args.curvesname+str(i),"r").read().splitlines())
        for j in range(len(lines[0])):
            color=next(COLORS)
            recordmeans = []
            for i in range(env_num):
                #color=next(COLORS)
                record, avgnum = [float(strs) for strs in lines[i][j].split("|")[:-1][:config_args.max_episodes]], 5 ###
                recordmean = [np.mean(record[k:k+avgnum]) for k in range(len(record)-avgnum+1)]
                recordmeans.append(recordmean)
                #plt.figure(112)
                #plt.plot(record,color=color,alpha=0.2)
                #plt.plot(recordmean,color=color,alpha=1.0)
            recordmeansmean = np.array(recordmeans).mean(axis=0)
            recordmeansvar = np.array(recordmeans).std(axis=0)
            plt.figure(113)
            plt.plot(recordmeansmean,color=color,alpha=0.6)
            plt.fill_between(range(len(recordmeansmean)), recordmeansmean-recordmeansvar, recordmeansmean+recordmeansvar,facecolor=color,alpha=0.2)
        plt.savefig(config_args.curvesname.replace('/','T')+'.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
if __name__ == '__main__':
    main()
