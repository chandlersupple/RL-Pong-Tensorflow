# Libraries
import gym
import time
import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()

# Returns a list of normalzied, discounted rewards
def process_rewards(rewards, gamma):
    processed_rewards = []
    for position, reward in enumerate(rewards): 
        processed_rewards.append(0)
        for k in range (position, len(rewards) - position):
            processed_rewards[-1] += np.power(gamma, k) * rewards[position + k]
    return (processed_rewards - np.mean(processed_rewards)) / np.std(processed_rewards)

init = tf.contrib.layers.xavier_initializer()
batch_size = 5000

# Placeholders
x = tf.placeholder(tf.float32, [None, 210, 160, 3])
a = tf.placeholder(tf.int32, [None])
R = tf.placeholder(tf.float32, [None])
x_flat = tf.layers.flatten(x)

# Policy Network
layer_one = tf.layers.dense(x_flat, 256, kernel_initializer= init)
act_one = tf.nn.leaky_relu(layer_one)
layer_two = tf.layers.dense(act_one, 5, kernel_initializer= init)
act_two = tf.nn.sigmoid(layer_two)
out = tf.multinomial(act_two, num_samples= 1)

# Optimization
loss = tf.reduce_sum(R * tf.losses.softmax_cross_entropy(tf.one_hot(a, 5), act_two))
train = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

# Training
env = gym.make('Pong-v0')
episode = 0

while True:
    current = env.reset()
    previous = 0
        
    observations = []
    actions = []
    rewards = []
    
    time_start = time.time()
    
    while len(observations) < batch_size:     
        env.render() # Renders environment (optional)
        
        observation = current - previous
        observations.append(observation)
        previous = current
        
        action = sess.run(out, feed_dict= {x: [observation]})[0][0] # Action outputted by policy network
        actions.append(action)
        
        current, reward, done, info = env.step(action)
        rewards.append(reward)
        
        if done: # Resets environment
            current = env.reset()
            previous = 0
            episode += 1 // 21
            
    # Trains network and provides training information
    sess.run(train, feed_dict= {x: observations, a: actions, R: process_rewards(rewards, 0.99)})
    print('Episode: %s, Completed in %s seconds' %(episode, round(time.time() - time_start, 2)))

env.close() # Closes environment when training is terminated
