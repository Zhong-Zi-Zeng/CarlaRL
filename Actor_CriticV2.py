import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from NetWorkV2 import ActorCriticNetwork


class Actor_Critic():
    def __init__(self,n_actions):
        self.n_action = n_actions
        self.lr = 0.0003
        self.gamma = 0.99

        self.actor_critic = ActorCriticNetwork(n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=self.lr))


    def choose_action(self,state):
        state = tf.convert_to_tensor([state],dtype=tf.float16)
        probability, _ = self.actor_critic(state)
        print(probability)
        action_prob = tfp.distributions.Categorical(probs=probability)
        action = action_prob.sample()

        return action.numpy()[0]

    def learn(self,state,action,reward,next_state,done):
        with tf.GradientTape() as tape:
            state = tf.convert_to_tensor([state], dtype=tf.float16)
            next_state = tf.convert_to_tensor([next_state], dtype=tf.float16)

            # critic網路更新
            _, state_value = self.actor_critic(state)
            _, next_state_value = self.actor_critic(next_state)

            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            TDtarget = reward + self.gamma * next_state_value * (1 - int(done))
            TDerror = TDtarget - state_value
            critic_loss = TDerror ** 2

            # actor網路更新
            probability, _ = self.actor_critic(state)
            action_prob = tfp.distributions.Categorical(probs=probability)
            actor_loss = -TDerror * action_prob.log_prob(action)

            total_loss = critic_loss + actor_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_weights)
        self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_weights))


    def save_model(self):
        self.actor_critic.save_weights('./model_weight/actor_critic')

    def load_model(self):
        self.actor_critic.load_weights("./model_weight/actor_critic")
