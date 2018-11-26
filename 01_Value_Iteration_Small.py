import numpy as np

def Utility(state, reward, transition_matrix, gamma, utility_vector, action_count):
     actions = np.zeros(action_count)
     for i in range(action_count):
          actions[i] = np.sum(np.multiply(utility_vector, np.dot(state, transition_matrix[:,:,i])))
     return reward + gamma*np.max(actions)

state_count = 17
direction_count = 2
gamma = .925
iter_count = 0
epsilon = 0.01

T = np.zeros((state_count,state_count,direction_count))
for i in range(state_count):
     for j in range(state_count):
          for k in range(direction_count):
               if i % 2 == 0 and j - i == 1 and k == 1:
                    T[i][j][k] = 0.05
               elif i % 2 == 0 and j - i == 2 and k == 1:
                    T[i][j][k] = 0.95
               elif i % 2 == 0 and j - i == 1 and k == 0:
                    T[i][j][k] = 1

rewards = np.array([ 0, 0, -150, 0, -2000, 0, -2000, 0, \
                     -2000, 0, -2000, 0, -2000, 0, -2000, 0, +20000])

utility_1 = np.zeros(state_count)

while True:
   delta = 0
   utility_0 = utility_1.copy()
   iter_count += 1
   for i in range(state_count):
       reward = rewards[i]
       state = np.zeros((1,state_count))
       state[0,i] = 1
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
                print "State #", j, ": ", utility_0[j]
           print("===================================================")
           break
