import time, os, tqdm, collections
import numpy as np
from trainer_base import BaseTrainer, EnvSummaryLogger
def linear(p):
    return 1-p
def constant(p):
    return 1
def lr_decay_parser(lr_decay):
    lr_decay_to_class = {'constant': constant, 'linear': linear}
    return lr_decay_to_class[lr_decay]
class Trainer(BaseTrainer):
    def __init__(self, sess, args, env, model):
        super().__init__(sess, args, env, model)
        self.test_timesteps   = args.test_timesteps
        self.train_iterations = int(args.train_iterations)
        self.print_iterations = int(self.train_iterations/args.print_iterations)#args.print_iterations
        self.save_iterations  = int(self.train_iterations/args.save_iterations)#args.save_iterations
        self.unroll_time_steps= args.unroll_time_steps
        self.learning_rate    = args.learning_rate
        self.lr_decay         = lr_decay_parser(args.lr_decay)
        self.gamma            = args.reward_discount_factor
        self.env_summary_logger = EnvSummaryLogger(sess, args.summary_dir, 'env', env.num_envs)
        self.train_input_shape= [args.unroll_time_steps*env.num_envs]+[args.num_stack]+list(env.observation_space.shape)
        self.observation_shape=                        [env.num_envs]+[args.num_stack]+list(env.observation_space.shape)
        self.observation_s = np.zeros(self.observation_shape, dtype=np.uint8)
        self.observation_s = self.__observation_update(self.env.reset(), self.observation_s)
    def __observation_update(self, new_observation, old_observation):# Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
        updated_observation = np.roll(old_observation, shift=-1, axis=1)    # updated_observation = env.num + model.input_shape = env.num + num_stack + env.obs.shape
        updated_observation[:,-1,:] = new_observation[:,:]                  # new_observation.shape = env.num + env.obs.shape
        return updated_observation
    def __discount_with_dones(self, rewards, dones, gamma):
        discounted, r = [], 0 # Start from downwards to upwards like Bellman backup operation.
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1.0 - done)
            discounted.append(r)
        return discounted[::-1]
    def __rollout(self):
        mb_observations, mb_actions, mb_values, mb_rewards, mb_dones = [], [], [], [], []
        for n in range(self.unroll_time_steps):
            actions, values, states           = self.model.step_policy.step(self.observation_s)
            mb_observations.append(np.copy(self.observation_s))
            mb_actions.append(actions)
            mb_values.append(values)
            observation, rewards, dones, info = self.env.step(actions) # plt.imsave(fname="img" + str(n) + ".png", arr=observation[0, :, :, 0], cmap='gray')
            mb_rewards.append(rewards)
            mb_dones.append(dones)
            for n, done in enumerate(dones):
                if done: self.observation_s[n] *= 0
            self.observation_s = self.__observation_update(observation, self.observation_s)
            self.env_summary_logger.add_summary_all(int(self.global_time_step_tensor.eval(self.sess)), info)
            self.global_time_step_assign_op.eval(session=self.sess, feed_dict={self.global_time_step_input: self.global_time_step_tensor.eval(self.sess)+1})
        # Conversion from (time_steps, num_envs) to (num_envs, time_steps)
        mb_observations = np.asarray(mb_observations,   dtype=np.uint8).swapaxes(1, 0).reshape(self.train_input_shape)
        mb_actions      = np.asarray(mb_actions,        dtype=np.int32).swapaxes(1, 0)
        mb_values       = np.asarray(mb_values,       dtype=np.float32).swapaxes(1, 0)
        mb_rewards      = np.asarray(mb_rewards,      dtype=np.float32).swapaxes(1, 0)
        mb_dones        = np.asarray(mb_dones,           dtype=np.bool).swapaxes(1, 0)
        last_values     = self.model.step_policy.value(self.observation_s).tolist()
        # Discount/bootstrap off value fn in all parallel environments
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards, dones = rewards.tolist(), dones.tolist()
            if dones[-1] == 0: rewards = self.__discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:              rewards = self.__discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        # Instead of (num_envs, time_steps). Make them num_envs*time_steps.
        mb_actions = mb_actions.flatten()
        mb_values  = mb_values.flatten()
        mb_rewards = mb_rewards.flatten()
        return mb_observations, mb_actions, mb_values, mb_rewards
    def __model_update(self, observations, actions, values, rewards, current_learning_rate):
        advantages = rewards - values
        feed_dict = {self.model.X_input: observations, self.model.actions: actions, self.model.rewards: rewards, self.model.advantages: advantages,
                     self.model.learning_rate: current_learning_rate}
        loss, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.model.loss, self.model.policy_gradient_loss, self.model.value_function_loss, self.model.entropy, self.model.optimize], feed_dict)
        return loss, policy_loss, value_loss, policy_entropy
    def train(self):
        print('Training...')
        tstart, loss_list, pe_list = time.time(), collections.deque(maxlen=self.print_iterations), collections.deque(maxlen=self.print_iterations)
        start_iteration = self.global_iteration_tensor.eval(self.sess)
        for iteration in tqdm.tqdm(range(start_iteration, self.train_iterations+1, 1), initial=start_iteration, total=self.train_iterations):
            observations, actions, values, rewards = self.__rollout()
            current_learning_rate = self.learning_rate * self.lr_decay(iteration/self.train_iterations)
            loss, policy_loss, value_loss, policy_entropy = self.__model_update(observations, actions, values, rewards, current_learning_rate)
            loss_list.append(loss)
            pe_list.append(policy_entropy)
            if not iteration % self.print_iterations:
                nseconds, tstart = time.time() - tstart, time.time()
                fps = int( (self.print_iterations*self.unroll_time_steps*self.env.num_envs) / nseconds )
                print('Iteration:'+str(iteration) + ' - loss: '+str(np.mean(loss_list))[:8] + ' - policy_entropy: '+str(np.mean(pe_list))[:8] + ' - fps: '+str(fps))
            if not iteration % self.save_iterations: self.save()
            self.global_iteration_assign_op.eval(session=self.sess, feed_dict={self.global_iteration_input: self.global_iteration_tensor.eval(self.sess)+1})
        self.env.close()
    def test(self):
        print('Testing...')
        for _ in tqdm.tqdm(range(self.test_timesteps)):
            actions, values, states        = self.model.step_policy.step(self.observation_s)
            observation, rewards, dones, _ = self.env.step(actions)
            for n, done in enumerate(dones): 
                if done: self.observation_s[n] *= 0
            self.observation_s = self.__observation_update(observation, self.observation_s)
        self.env.close()
