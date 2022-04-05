import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from NetWork import CriticNetwork,ActorNetwork



class Actor_Critic():
    def __init__(self,n_actions):
        self.n_action = n_actions
        self.lr = 0.0005
        self.beta = 0.001
        self.gamma = 0.99
        self.TDerror = None

        self.critic_network = CriticNetwork()
        self.actor_network = ActorNetwork(n_actions)

        self.critic_network.compile(optimizer=Adam(learning_rate=self.beta))
        self.actor_network.compile(optimizer=Adam(learning_rate=self.lr))


    def choose_action(self,state):
        state = tf.convert_to_tensor([state],dtype=tf.float16)
        probability = self.actor_network(state)
        action_probs = tfp.distributions.Categorical(probs=probability)
        action = action_probs.sample()

        return action.numpy()[0]

    def learn_critic(self,state,reward,next_state,done):
        with tf.GradientTape() as tape:
            state = tf.convert_to_tensor([state], dtype=tf.float16)
            next_state = tf.convert_to_tensor([next_state], dtype=tf.float16)

            state_value = self.critic_network(state)
            next_state_value = self.critic_network(next_state)

            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            TDtarget = reward + self.gamma * next_state_value * (1-int(done))
            self.TDerror = TDtarget - state_value
            Loss = self.TDerror ** 2

        gradient = tape.gradient(Loss,self.critic_network.trainable_weights)
        self.critic_network.optimizer.apply_gradients(zip(gradient,self.critic_network.trainable_weights))


    def learn_actor(self,state,action):
        with tf.GradientTape() as tape:
            state = tf.convert_to_tensor([state],dtype=tf.float16)
            probability = self.actor_network(state)
            action_probs = tfp.distributions.Categorical(probs=probability)
            Loss = -self.TDerror * action_probs.log_prob(action)

        gradient = tape.gradient(Loss,self.actor_network.trainable_weights)
        self.actor_network.optimizer.apply_gradients(zip(gradient, self.actor_network.trainable_weights))


    def save_model(self):
        self.actor_network.save_weights('./model_weight/actor_network')
        self.critic_network.save_weights('./model_weight/critic_network')

    def load_model(self):
        self.actor_network.load_weights("./model_weight/actor_network")
