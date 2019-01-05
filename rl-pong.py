# Libraries
import gym
import time
import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()

# Returns a list of normalzied, discounted rewards
def discount_rewards(rewards, discount_factor):
    discounted_rewards = []
    for position in range (len(rewards)):
        gamma = 1
        reward_sum = 0
        for k in range (len(rewards) - position):
            reward_sum += gamma * rewards[position + k] 
            gamma *= discount_factor
            if rewards[k] != 0:
                break
        discounted_rewards.append(reward_sum)
    return (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards) # Normalizes discounted rewards

def preprocess(observation):
    observation = observation[34:193, :, 0]
    return observation[::2, ::2]
    
init = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer(scale= 0.1)
batch_size = 20
learning_rate = 0.0001

# Placeholders
x = tf.placeholder(tf.float32, [None, 80 * 80])
action = tf.placeholder(tf.int32, [None])
reward = tf.placeholder(tf.float32, [None])

# Policy Network
layer_one = tf.layers.dense(x, 256, kernel_initializer= init, kernel_regularizer= regularizer)
act_one = tf.nn.leaky_relu(layer_one)
layer_two = tf.layers.dense(act_one, 1, kernel_initializer= init, kernel_regularizer= regularizer)
act_two = tf.nn.sigmoid(layer_two)

# Optimization
loss = tf.losses.softmax_cross_entropy(onehot_labels= tf.one_hot(action, 1), logits= layer_two, weights= reward)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

# Training
env = gym.make('Pong-v0')
episodes = 0

while True:
    current = env.reset()
    previous = current
    score = 0
    
    observations = []
    actions = []
    rewards = []
    
    time_start = time.time()
    
    for episode in range (batch_size):
        while True:
            env.render() # Renders environment (optional)
            observation = np.reshape(preprocess(current - previous), [80 * 80])
            observations.append(observation)
            previous = current
            
            if sess.run(act_two, feed_dict= {x: [observation]}) > np.random.uniform(): direction = 2 # Action outputted by policy network
            else: direction = 3
            actions.append(direction)
            
            current, game_reward, done, info = env.step(direction)
            if game_reward > 0: score += 1
            rewards.append(game_reward)
            
            if done: # Resets environment
                current = env.reset()
                previous = current
                break
            
    # Trains network and provides training information
    episodes += batch_size
    
    sess.run(train, feed_dict= {x: observations, action: actions, reward: discount_rewards(rewards, 0.99)})
    print('Episode %s, Completed in %s seconds, Average Score: %s' %(episodes, round(time.time() - time_start, 2), np.round(score / batch_size, 2)))
    
    if episodes % 500 == 0: saver.save(sess, 'save-dir')
    
env.close() # Closes environment when training is terminated
