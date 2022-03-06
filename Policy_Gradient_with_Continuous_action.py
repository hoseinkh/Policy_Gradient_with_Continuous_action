###############################################################################
# For more info, see https://hoseinkh.github.io/
###############################################################################
import gym
import os
import sys
import numpy as np
"""
# if using tensorflow v1:
import tensorflow as tf
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import matplotlib
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from gym import wrappers
from datetime import datetime
###############################################################################
# Feature transformer uses RBF kernels to transform the original state space to ...
# ... higher dimensions. This helps with the performance of the model!
class FeatureTransformer:
  def __init__(self, env, n_components=500):
    # generate states (observations)
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    # define scaler and scale the states (observations) --> mean 0 and variance 1
    scaler = StandardScaler()
    scaler.fit(observation_examples)
    #
    # Now we basically use RBF to for feature generation
    # Each RBFSampler takes each (original) (feature representation) of ...
    # ... a state and converts it to "n_components" new featuers.
    # Hence, after concatenating the new features, we convert each state to ...
    # ... {(# RBF samplers) * n_components} new features.
    #
    # We use RBF kernels with different variances to cover different parts ...
    # ... of the space.
    #
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    # For all the generated samples, transform original state representaions ...
    # ... to a new state representation using "featurizer"
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))
    #
    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer
  ######################################
  def transform(self, observations):
    #
    scaled_original_state_representation = self.scaler.transform(observations)
    #
    scaled_higher_dimensions_state_representation = self.featurizer.transform(scaled_original_state_representation)
    return scaled_higher_dimensions_state_representation
###############################################################################
# It is better to define everything directly. This allows tensorflow to ...
# ... automatically calculate the cost functions, and hence we get rid of ...
# ... the issue of manually feeding it to the tensorflow.
# To do this TensorFlow needs to remember what operations happen in what ...
# ... order during the forward pass. Then, during the backward pass, ...
# ... TensorFlow traverses this list of operations in reverse order to ...
# ... compute gradients.
class HiddenLayer:
  def __init__(self, inp_size_of_hidden_layer, out_size_of_hidden_layer, f=tf.nn.tanh, use_bias=True, zeros=False):
    if zeros:
      W = np.zeros((inp_size_of_hidden_layer, out_size_of_hidden_layer), dtype=np.float32)
    else:
      W = tf.random_normal(shape=(inp_size_of_hidden_layer, out_size_of_hidden_layer)) * np.sqrt(2. / inp_size_of_hidden_layer, dtype=np.float32)
    self.W = tf.Variable(W)
    #
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(out_size_of_hidden_layer).astype(np.float32))
    #
    self.f = f
  ######################################
  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)
###############################################################################
# approximates pi(a | s)
# here we use two NNs. One for predicting the mean of the action, and one to ...
# ... predict the std of the action. However, the two NNs have the same body, ...
# ... and only the last layer differs!
class PolicyModel:
  def __init__(self, data_input_size, feature_transformer, hidden_layer_sizes=[]):
    self.feature_transformer = feature_transformer
    #
    ##### hidden layers #####
    NN_input_size = data_input_size
    self.hidden_layers = []
    for NN_output_size in hidden_layer_sizes:
      layer = HiddenLayer(NN_input_size, NN_output_size)
      self.hidden_layers.append(layer)
      NN_input_size = NN_output_size
    #
    ## final layer for the mean (we use linear for the activation function)
    self.mean_layer = HiddenLayer(data_input_size, 1, lambda x: x, use_bias=False, zeros=True)
    #
    ## final layer for the variance (we use softplus for the activation function to ensure positive std)
    self.stdv_layer = HiddenLayer(data_input_size, 1, tf.nn.softplus, use_bias=False, zeros=False)
    #
    ### inputs and targets (used in the session)
    ## self.X is the feature representaion of the state (after applying self.feature_transformer)
    self.X = tf.placeholder(tf.float32, shape=(None, data_input_size), name='X')
    self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
    ## self.advantages is the G - V(S), which uses V(S) as a Baseline to ...
    ## ... decrease variance of the model!
    self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
    #
    ### get final hidden layer
    out_of_curr_layer = self.X
    for layer in self.hidden_layers:
      out_of_curr_layer = layer.forward(out_of_curr_layer)
    #
    ### calculate output and cost
    ## calculate the mean of the Gaussian distribution for the action
    mean = self.mean_layer.forward(out_of_curr_layer)
    ## calculate the std of the Gaussian distribution for the action
    stdv = self.stdv_layer.forward(out_of_curr_layer) + 1e-5 # we do smoothing by adding small amount to the std
    #
    ### make mean and std 1-D
    mean = tf.reshape(mean, [-1])
    stdv = tf.reshape(stdv, [-1]) 
    #
    ### Build the normal distribution of the action
    norm = tf.distributions.Normal(mean, stdv)
    ## note that the actions in the environment are between -1 and 1
    self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)
    #
    log_probs = norm.log_prob(self.actions)
    ## note that here we add a regularization term (i.e. 0.1*norm.entropy()) to the cost function ...
    ## ... to avoid overfitting!
    cost = -tf.reduce_sum(self.advantages * log_probs + 0.1*norm.entropy())
    self.train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)
  ######################################
  def set_session(self, session):
    self.session = session
  ######################################
  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    X = self.feature_transformer.transform(X)
    #
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: X,
        self.actions: actions,
        self.advantages: advantages,
      }
    )
  ######################################
  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.feature_transformer.transform(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})
  ######################################
  def sample_action(self, X):
    p = self.predict(X)[0]
    return p
###############################################################################
# approximates V(s)
# we use this function to calculate state-value function V(s) ...
# ... which is used as Baseline in the policy gradient, which ...
# ... helps decreasing the variance of the model!
class ValueModel:
  def __init__(self, data_input_size, feature_transformer, hidden_layer_sizes=[]):
    self.feature_transformer = feature_transformer
    self.costs = []
    #
    # create the neural network for the state-value approximation (i.e. V(S))
    self.layers = []
    NN_input_size = data_input_size
    for NN_output_size in hidden_layer_sizes:
      layer = HiddenLayer(NN_input_size, NN_output_size)
      self.layers.append(layer)
      NN_input_size = NN_output_size
    #
    ## final layer. Since we are predicting the value function, we only have one node, and ...
    ## ... the linear function is used as the activation function in the output layer
    layer = HiddenLayer(NN_input_size, 1, lambda x: x)
    self.layers.append(layer)
    #
    ### inputs and targets
    ## self.X is the (feature-transformed) feature representation of the state
    self.X = tf.placeholder(tf.float32, shape=(None, data_input_size), name='X')
    ## self.Y is the observed value for the state S.
    self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')
    #
    ### calculate output and cost
    out_of_curr_layer = self.X # = feature representation of the state
    for layer in self.layers:
      out_of_curr_layer = layer.forward(out_of_curr_layer)
    Y_hat = tf.reshape(out_of_curr_layer, [-1]) # the output of the NN (estimated V(s))
    self.predict_op = Y_hat
    #
    ### we use the squared error as the error function!
    cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
    self.cost = cost
    self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)
  ######################################
  def set_session(self, session):
    self.session = session
  ######################################
  def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    X = self.feature_transformer.transform(X)
    Y = np.atleast_1d(Y)
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
    cost = self.session.run(self.cost, feed_dict={self.X: X, self.Y: Y})
    self.costs.append(cost)
  ######################################
  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.feature_transformer.transform(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})
###############################################################################
def play_one_td(env, policy_model, value_model, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  #
  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = policy_model.sample_action(observation)
    prev_observation = observation
    observation, reward, done, info = env.step([action])
    #
    totalreward += reward
    #
    # update the models
    V_next = value_model.predict(observation)
    G = reward + gamma*V_next
    advantage = G - value_model.predict(prev_observation)
    policy_model.partial_fit(prev_observation, action, advantage)
    value_model.partial_fit(prev_observation, G)
    #
    iters += 1
  #
  return totalreward, iters
###############################################################################
# we are evaluating the performance of the model at each time t by ...
# ... taking the running average of the adjacent 100 iterations to that time t.
def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.xlabel("Iterations")
  plt.ylabel("Average Time")
  # plt.show()
  curr_path = os.path.abspath(os.getcwd())
  plt.savefig(curr_path + '/figs/reward_running_avg_MountainCarContinuous.png')
  plt.close()
###############################################################################
# here we plot the negative of the optimal state value functions (i,e, -V*(s))!
# Note that the optimal action values are equal to the negative of the average optimal time ...
# ... that it takes to reach the mountain.
# Hence this plot shows the average optimal time to reach the top of the mountain at each state.
def plot_avg_num_remaining_steps(env, estimator, num_tiles=20):
  x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
  y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
  X, Y = np.meshgrid(x, y)
  # both X and Y will be of shape (num_tiles, num_tiles)
  Z = np.apply_along_axis(lambda _: -1*np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
  # Z will also be of shape (num_tiles, num_tiles)
  #
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
  ax.set_xlabel('Position')
  ax.set_ylabel('Velocity')
  ax.set_zlabel('Num steps to reach mountain == -V(s)')
  ax.set_title("Num steps to Reach Mountain Function")
  fig.colorbar(surf)
  fig.savefig("./figs/Num_steps_to_Reach_Mountain.png")
  # plt.show()
  plt.close()
###############################################################################
if __name__ == '__main__':
  env = gym.make('MountainCarContinuous-v0').env
  feature_transformer = FeatureTransformer(env, n_components=100)
  D = feature_transformer.dimensions
  policy_model = PolicyModel(D, feature_transformer, [])
  value_model = ValueModel(D, feature_transformer, [])
  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)
  policy_model.set_session(session)
  value_model.set_session(session)
  discount_rate = 0.95
  #
  if True:
    monitor_dir = os.getcwd() + "/videos/" + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
  #
  N = 50
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    totalreward, num_steps = play_one_td(env, policy_model, value_model, discount_rate)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n, "total reward: %.1f" % totalreward, "num steps: %d" % num_steps, "avg reward (last 100): %.1f" % totalrewards[max(0, n-100):(n+1)].mean())
  #
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  #
  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.savefig("./figs/reward_avg_MountainCarContinuous_policy_gradient_continuous_action.png")
  plt.show()
  plt.close()
  #
  plot_running_avg(totalrewards)
  plot_avg_num_remaining_steps(env, value_model)


