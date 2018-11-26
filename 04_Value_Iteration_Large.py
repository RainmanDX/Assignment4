import numpy as np

def Utility(state, reward, transition_matrix, gamma, utility_vector, action_count):
     actions = np.zeros(action_count)
     for i in range(action_count):
          actions[i] = np.sum(np.multiply(utility_vector, np.dot(state, transition_matrix[:,:,i])))
     return reward + gamma*np.max(actions)

state_count = 58
direction_count = 5
gamma = .92
iter_count = 0
epsilon = 0.01

T = np.load("T_large.npy")

rewards = np.array([
     0, 0, -2, -1, 0, 0, -1, -1, -0.5, 0, \
     -2, -1, -1, -2, -1, -1, -1, -0.5, -1, -1, \
     -1, -0.5, 0, -1, -0.5, -0.5, -0.5, 0, 0, -0.5, \
     -1, -1, 0, 0, -0.5, -1, -0.5, -1, -2, -1, \
     -1, -1.5, -0.5, -1, -1, -1.5, -1.5, 0, \
     0, 0, -1, -1, -1, 0, -0.5, -2, -10, +100])

utility_1 = np.zeros(state_count)

while True:
   delta = 0
   utility_0 = utility_1.copy()
   iter_count += 1
   for i in range(state_count):
       reward = rewards[i]
       state = np.zeros((1,state_count))
       state[0,i] = 1
#       print state
#       print(T.shape)
       utility_1[i] = Utility(state, reward, T, gamma, utility_0, direction_count)
       delta = max(delta, np.abs(utility_1[i] - utility_0[i]))
   if delta < epsilon * (1 - gamma) / gamma:
           print("=================== FINAL RESULT ==================")
           print("Iterations: " + str(iter_count))
           print("Delta: " + str(delta))
           print("Gamma: " + str(gamma))
           print("Epsilon: " + str(epsilon))
           print("===================================================")
           for j in range(state_count):
                print "State #", j, ": ", round(utility_0[j], 1)
           print("===================================================")
           break
