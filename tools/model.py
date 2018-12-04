import tensorflow as tf
import numpy as np
from model_layers import mse, openai_entropy
from model_layers import conv2d, flatten, dense
from model_layers import orthogonal_initializer, noise_and_argmax
class CNNPolicy:
    def __init__(self, args, sess, X_input, num_actions, reuse=False, is_training=False):
        self.sess = sess
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
            #self.conv1 = conv2d('conv1', self.cast, num_filters=32, kernel_size=(8, 8), padding='VALID', stride=(4, 4),
            #            initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu, is_training=is_training)
            #self.conv2 = conv2d('conv2', self.cast, num_filters=self.convs[0][0], kernel_size=(self.convs[0][1], self.convs[0][2]), padding='VALID', stride=(1, 1),#conv1 64
            #            initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu, is_training=is_training)
            #self.conv3 = conv2d('conv3', self.conv2, num_filters=self.convs[1][0], kernel_size=(self.convs[1][1], self.convs[1][2]), padding='VALID', stride=(1, 1),#64 3, 3
            #            initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu, is_training=is_training)
            #self.conv_flattened = flatten(self.conv3)
            self.conv_flattened = flatten(self.conv)
            self.dens = self.conv_flattened
            for i,dens in enumerate(self.denss):
                densout = dense('dens'+str(i), self.dens, output_dim=dens,
                        initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu, is_training=is_training)
                self.dens = densout
                self.densouts.append(densout)
            #self.fc4 = dense('fc4', self.conv_flattened, output_dim=args.densedim, initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu, is_training=is_training)
            self.policy_logits  = dense('policy_logits',  self.dens, output_dim=num_actions, initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=is_training)
            self.value_function = dense('value_function', self.dens, output_dim=1          , initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=is_training)
            with tf.name_scope('action'):
                self.action_s = noise_and_argmax(self.policy_logits)
            with tf.name_scope('value'):
                self.value_s = self.value_function[:, 0]
    def step(self, observation, *_args, **_kwargs):
        #self.debug(observation)
        action, value = self.sess.run([self.action_s, self.value_s], {self.P_input: observation})
        return action, value, []  # dummy state
    def value(self, observation, *_args, **_kwargs):
        return self.sess.run(self.value_s, {self.P_input: observation})
    def debug(self, observation, *_args, **_kwargs):
        print(observation.shape)
        cast = self.sess.run([self.cast], {self.P_input: observation})
        print('cast',cast.shape)
        for i in range(len(self.convouts)):
            convi = self.sess.run([self.convouts[i]], {self.P_input: observation})
            print('conv'+str(i),convi.shape)
        conv_flattened = self.sess.run([self.conv_flattened], {self.P_input: observation})
        print('conv_flattened',conv_flattened.shape)
        for i in range(len(self.densouts)):
            densi = self.sess.run([self.densouts[i]], {self.P_input: observation})
            print('dens'+str(i),densi.shape)
        policy_logits, value_function, action_s, value_s = self.sess.run([self.policy_logits, self.value_function, self.action_s, self.value_s], {self.P_input: observation})
        print('policy_logits',policy_logits.shape)
        print('value_function',value_function.shape)
        print('action_s',action_s.shape)
        print('value_s',value_s.shape)
        exit()
def policy_name_parser(policy_name):
    policy_to_class = {'CNNPolicy': CNNPolicy}#, 'MNNPolicy': MNNPolicy}
    return policy_to_class[policy_name]
class Model:
    def __init__(self, sess, args, observation_space_params, num_actions):
        self.sess = sess
        self.input_shape   = [args.num_stack]+list(observation_space_params)
        self.num_actions   = num_actions
        self.policy        = policy_name_parser(args.policy_name)
        self.entropy_coeff = args.entropy_coef
        self.valuefc_coeff = args.value_function_coeff
        self.max_grad_norm = args.max_gradient_norm
        #self.alpha   = args.alpha # RMSProp params = {'learning_rate': 7e-4, 'alpha': 0.99, 'epsilon': 1e-5}
        #self.epsilon = args.epsilon
        with tf.name_scope('train_input'):
            self.X_input        = tf.placeholder(tf.uint8, [None]+self.input_shape)
            self.actions        = tf.placeholder(tf.int32, [None])
            self.rewards        = tf.placeholder(tf.float32, [None])
            self.advantages     = tf.placeholder(tf.float32, [None])
            self.learning_rate  = tf.placeholder(tf.float32, [])
        self.step_policy  = self.policy(args, self.sess, self.X_input, self.num_actions, reuse=False, is_training=False)
        self.train_policy = self.policy(args, self.sess, self.X_input, self.num_actions, reuse=True , is_training=True)
        with tf.variable_scope('train'):
            negative_log_prob_action  = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.train_policy.policy_logits, labels=self.actions)
            self.policy_gradient_loss = tf.reduce_mean(self.advantages * negative_log_prob_action)
            self.value_function_loss  = tf.reduce_mean(mse(tf.squeeze(self.train_policy.value_function), self.rewards))
            self.entropy              = tf.reduce_mean(openai_entropy(self.train_policy.policy_logits))
            self.loss = self.policy_gradient_loss - self.entropy*self.entropy_coeff + self.value_function_loss*self.valuefc_coeff
            with tf.variable_scope("policy"):
                params = tf.trainable_variables()
                grads  = tf.gradients(self.loss, params)
                if self.max_grad_norm is not None: grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads  = list(zip(grads, params))
                #self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.alpha, epsilon=self.epsilon).apply_gradients(grads)
                self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(grads)
