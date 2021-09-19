import numpy as np
from sklearn.linear_model import LinearRegression


class AgentBilinear():
    """
    This is an attempt to reproduce the results of the Mountain car experiments mentioned in the paper on Reinforcement
    learning with a bilinear Q function. The game is implemented in openai gym found here:
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
    """
    def __init__(self, env, expanded_action_space_size=2, expanded_state_space_size=6,
    				max_action=0.001, min_action=-0.001, discount_rate=0.9) -> None:
        self.env = env
        self.max_action = max_action
        self.min_action = min_action
        self.expanded_action_space_size = expanded_action_space_size
        self.expanded_state_space_size = expanded_state_space_size
        self.W = np.random.rand(self.expanded_state_space_size, self.expanded_action_space_size)
        self.discount_rate = discount_rate

    def expand_state_space(self, s:np.array) -> np.array:
        """
        This function expands the state space by adding more state representations to the state vector
        ----------------
        Input:
            s: np.array (x,2)

        Output:
            expanded_state: np.array (x,expanded_state_space_size)
        """
        num_dim = s.ndim
        if num_dim != 2:
            raise Exception('Number of input dimensions must be equal to 2.')
        num_rows = s.shape[0]
        expanded_state = np.zeros((num_rows, self.expanded_state_space_size))

        expanded_state[:,:2] = s
        expanded_state[:,2] = s[:,0]**2
        expanded_state[:,3] = s[:,0] * s[:,1]
        expanded_state[:,4] = s[:,1]**2
        expanded_state[:,5] = s[:,0]**3

        return expanded_state

    def expand_action_space(self, a) -> np.array:
        """
        This function expands the action space by adding a pseudo action with magnitude 1 to the action vector
        ----------------
        Input:
            a: np.array (x, 1)

        Output:
            expanded_action: np.array (x, expanded_action_space_size)
        """
        num_dim = a.ndim
        if num_dim != 2:
            raise Exception('Number of input dimensions must be equal to 2.')
        num_rows = a.shape[0]
        expanded_action = np.zeros((num_rows, self.expanded_action_space_size))

        expanded_action[:,0] = np.ones(num_rows)
        expanded_action[:,1] = a[:,0]

        return expanded_action

    def step(self, s, a):
        """
        In principle taken from the step function provided in the openai gym Mountain Car Continuous repo. 
        The difference is this custom function takes in both a state and an action to compute the next state.
        ----------------
        Input:
            s: np.array (x, 2)
            a: np.array (x, 1)

        output:
            reward: float
            s_prime: np.array (x, 2)
        """
        power = 0.0015
        max_speed = 0.07
        min_position = -1.2
        max_position = 0.6
        goal_position = 0.45
        goal_velocity = 0

        position = np.copy(s[:,0])
        velocity = np.copy(s[:,1])
        force = np.copy(a[:,-1])

        velocity += force * power - 0.0025 * np.cos(3 * position)
        velocity[velocity > max_speed] = max_speed
        velocity[velocity < -max_speed] = -max_speed
        position += velocity
        position[position > max_position] = max_position
        position[position < min_position] = min_position
        velocity[np.where(((position == min_position) & (velocity < 0)))] = 0

        done = np.where(((position >= goal_position) & (velocity >= goal_velocity)))
        not_done = np.where(((position < goal_position) | (velocity < goal_velocity)))

        reward = np.zeros((position.shape[0], 1))
        reward[done] = 100
        reward[not_done, -1] -= np.power(a[not_done, -1], 2) * 0.1

        num_states = len(position)
        s_prime = np.zeros((num_states, 2))
        s_prime[:,0] = position
        s_prime[:,1] = velocity

        return reward, s_prime

    def make_replay_buffer(self, num_examples: int) -> None:
        """
        Takes in the desired number of examples to make the replay buffer from (basically the size of the replay buffer).
        Saves a list of tuples of the form <s, a, r, s'> in the variable replay_buffer.
        ----------------
        Input:
            num_examples: int

        output:
            replay_buffer: List[Tuple[state, action, reward, next state]]
        """
        observation_examples = np.array([self.env.observation_space.sample() for _ in range(num_examples)])
        observation_examples = self.expand_state_space(observation_examples)
        action_examples = np.array([self.env.action_space.sample() for _ in range(num_examples)])
        action_examples[action_examples < self.min_action] = self.min_action
        action_examples[action_examples > self.max_action] = self.max_action
        action_examples = self.expand_action_space(action_examples)
        reward, s_prime = self.step(observation_examples, action_examples)

        replay_buffer = [(observation_examples[i], action_examples[i], reward[i], s_prime[i]) for i in range(num_examples)]
        return replay_buffer

    def make_targets(self, replay_buffer):
        """
        Prepares the targets for the linear regression
        ----------------
        Input:
            replay_buffer: List[Tuple[state, action, reward, next state]]

        output:
            targets: np.array (len(replay_buffer), 1)
        """
        num_examples = len(replay_buffer)
        targets = np.zeros((num_examples, 1))
        for i in range(num_examples):
            _, _, r, s_prime = replay_buffer[i]
            a = self.get_action(s_prime)
            targets[i,-1] = r + self.discount_rate * self.get_Qsa(s_prime, a)
        return targets

    def get_Qsa(self, s:np.array, a:np.array) -> float:
        """
        Returns the Q-value given a state and an action
        ----------------
        Input:
            s: np.array
            a: np.array

        output:
            sWa: float
        """
        if s.shape != (1, self.expanded_state_space_size):
            temp = np.zeros((1, s.shape[0]))
            temp[0,:] = s
            s = temp
            s = self.expand_state_space(s)
        if a.shape != (1, self.expanded_action_space_size):
            temp = np.zeros((1, a.shape[0]))
            temp[0,-1] = a
            a = temp
            a = self.expand_action_space(a)
        sW = np.matmul(s, self.W)
        sWa = np.matmul(sW, a.T)
        return sWa[0,0]
            

    def make_training_set(self, replay_buffer):
        """
        Prepares the training set for the linear regression
        ----------------
        Input:
            replay_buffer: List[Tuple[state, action, reward, next state]]

        output:
            X_train: np.array (len(replay_buffer), self.expanded_state_space_size*self.expanded_action_space_size)
        """
        num_examples = len(replay_buffer)
        X_train = np.zeros((num_examples, self.expanded_state_space_size*self.expanded_action_space_size))
        for i in range(num_examples):
            s, a, _, _ = replay_buffer[i]
            temp = np.zeros((1, s.shape[0]))
            temp[0,:] = s
            s = temp
            temp = np.zeros((a.shape[0],1))
            temp[:,0] = a
            a = temp
            x = np.matmul(a, s)
            x = np.append(x[0,:], x[1,:])
            X_train[i,:] = x
        return X_train

    def train(self, num_examples, horizon=None):
        """
        Does a linear regression to produce new W
        """
        W_old = -np.copy(self.W)
        replay_buffer = self.make_replay_buffer(num_examples)
        X_train = self.make_training_set(replay_buffer)
        if horizon == None:
            while True:
                targets = self.make_targets(replay_buffer)
                regr = LinearRegression()
                regr.fit(X_train, targets)
                self.W = np.reshape(regr.coef_, (self.W.shape), order='F')
                stop_condition = np.sum(np.abs(self.W-W_old)) < 0.01
                if stop_condition:
                    break
                W_old = np.copy(self.W)
        else:
            for _ in range(horizon):
                targets = self.make_targets(replay_buffer)
                regr = LinearRegression()
                regr.fit(X_train, targets)
                self.W = np.reshape(regr.coef_, (self.W.shape), order='F')
                W_old = np.copy(self.W)

    def get_action(self, s: np.array) -> float:
        """
        Given a state, return the *optimal* action.
        ----------------
        Input:
            s: np.array

        output:
            a: np.array
        """
        if s.shape != (1, self.expanded_state_space_size):
            temp = np.zeros((1, s.shape[0]))
            temp[0,:] = s
            s = temp
            s = self.expand_state_space(s)
        x = np.matmul(s, self.W)
        slope = x[0,1]
        a = 0
        if slope < 0: a = self.min_action
        elif slope > 0: a = self.max_action
        return np.array([a])
