import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

"""建立神經網路"""
def build_dqn(lr,n_actions,**kwargs):
    model = Sequential()

    model.add(BatchNormalization())
    model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='relu6'))
    model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='relu6'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu6'))
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu6'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu6'))
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu6'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(GlobalMaxPooling2D())
    model.add(Dense(n_actions))

    model.build(input_shape=(1,kwargs['img_height'],kwargs['img_width'],3))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=lr),loss='mse')

    return model

"""暫存器設置"""
class ReplayBuffer():
    def __init__(self,max_mem,n_action,**kwargs):
        self.max_mem = max_mem
        self.state_memory = np.zeros((self.max_mem,kwargs['img_height'],kwargs['img_width'],3),dtype=np.float32)
        self.next_state_memory = np.zeros((self.max_mem,kwargs['img_height'],kwargs['img_width'],3),dtype=np.float32)
        self.action_memory = np.zeros((self.max_mem,n_action),dtype=np.int8)
        self.reward_memory = np.zeros(self.max_mem)
        self.terminal_memory = np.zeros(self.max_mem,dtype=np.float32)
        self.mem_counter = 0

    def store_transition(self,state,action,reward,next_state,done):
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1
        self.action_memory[self.mem_counter] = actions
        self.state_memory[self.mem_counter] = state
        self.reward_memory[self.mem_counter] = reward
        self.next_state_memory[self.mem_counter] = next_state
        self.terminal_memory[self.mem_counter] = 1 - int(done)

        self.mem_counter += 1
        if(self.mem_counter == self.max_mem):
            self.mem_counter = 0

    """從memory隨機抽取mini_batch的資料"""
    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_counter,self.max_mem)
        batch = np.random.choice(max_mem,batch_size)

        state = self.state_memory[batch]
        next_state = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return state, actions, rewards, next_state, terminal


class Agent():
    def __init__(self
                 ,lr
                 ,gamma
                 ,n_actions
                 ,epsilon
                 ,batch_size
                 ,epsilon_end
                 ,mem_size
                 ,epsilon_dec
                 ,img_width
                 ,img_height
                 ,fixed_q=False
                 ,iteration=200
                 ,f_name="dqn_model.h5"):

        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.img_width = img_width
        self.img_height = img_height
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.model_file = f_name
        """建置資料庫"""
        self.memory = ReplayBuffer(max_mem=mem_size,n_action=n_actions,img_width=self.img_width,img_height=self.img_height)
        """建置模型"""
        self.q_eval = build_dqn(lr=self.lr,n_actions=self.n_actions,img_width=self.img_width,img_height=self.img_height)
        """是否使用target_network"""
        self.fixed_q = fixed_q
        self.fixed_q_al(iteration)

    """使用fixed-q演算法"""
    def fixed_q_al(self,iteration):
        if (self.fixed_q):
            self.iteration = iteration
            self.iteration_counter = 0
            self.q_target_net = build_dqn(lr=self.lr,n_actions=self.n_actions,img_width=self.img_width,img_height=self.img_height)
            self.q_target_net.set_weights(self.q_eval.get_weights())

    def remember(self,state,action,reward,next_state,done):
        self.memory.store_transition(state,action,reward,next_state,done)

    def choose_action(self,state):
        state = state[np.newaxis,:]
        rand = np.random.random()

        if(rand < self.epsilon):
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if(self.memory.mem_counter < self.batch_size):
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space,dtype=np.int8)
        action_indices = np.dot(action,action_values)

        """每batch_size次後下降epsilon"""
        if(self.memory.mem_counter % self.batch_size == 0):
            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end

        """使用fixed-q算法"""
        if(self.fixed_q):
            # 新神經網路預測的Q值
            q_eval = self.q_eval.predict(state)
            # 舊神經網路預測的Q值
            q_target_pre = self.q_target_net.predict(next_state)

            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            # 貝爾曼方程
            q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_target_pre, axis=1) * done
            # 更新參數
            self.q_eval.fit(state, q_target, verbose=0)

            """到達指定迭代次數後，複製權重給q_target_net"""
            self.iteration_counter += 1
            if(self.iteration_counter == self.iteration):
                self.q_target_net.set_weights(self.q_eval.get_weights())
                self.iteration_counter = 0
        else:
            q_eval = self.q_eval.predict(state)
            q_next = self.q_eval.predict(next_state)

            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done

            _ = self.q_eval.fit(state, q_target, verbose=0)

    """儲存模型"""
    def save_model(self):
        if(self.fixed_q):
            self.q_target_net.save(self.model_file)
        else:
            self.q_eval.save(self.model_file)

    """載入模型"""
    def load_model(self):
        self.q_eval = load_model(self.model_file)


