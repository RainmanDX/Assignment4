import numpy as np

def Utility(policy, rewards, transition_matrix, gamma, utility_vector, state_count):
     for index in range(state_count):
          if not np.isnan(policy[index]):
               state = np.zeros((1,state_count))
               state[0,index] = 1
               action = int(policy[index])
               utility_vector[index] = rewards[index] + gamma*np.sum(np.multiply(utility_vector, np.dot(state, transition_matrix[:,:,action])))
     return utility_vector

def Action(utility_vector, transition_matrix, state, action_count):
     actions = np.zeros(action_count)
     for i in range(action_count):
          actions[i] = np.sum(np.multiply(utility_vector, np.dot(state, transition_matrix[:,:,i])))
     return np.argmax(actions)

def display(p, shape):
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(p[counter] == -1): policy_string += " %  "            
            elif(p[counter] == 0): policy_string += " No  "
            elif(p[counter] == 1): policy_string += " Yes  "           
            elif(np.isnan(p[counter])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)

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

policy = np.random.randint(0, direction_count, size=(state_count)).astype(np.float32)
for state_index in range(len(policy)):
     if state_index % 2 == 1 or state_index == 16:
          policy[state_index] = -1

rewards = np.array([ 0, 0, -150, 0, -2000, 0, -2000, 0, \
                     -2000, 0, -2000, 0, -2000, 0, -2000, 0, +20000])

utility = np.zeros(state_count)

while True:
     iter_count += 1
     utility_0 = utility.copy()
     utility = Utility(policy, rewards, T, gamma, utility, state_count)
     delta = np.absolute(utility - utility_0).max()
     if delta < epsilon*(1 - gamma) / gamma:
          break
     for index in range(state_count):
          if not np.isnan(policy[index]) and not policy[index] == -1:
               state = np.zeros((1, state_count))
               state[0,index] = 1
               a = Action(utility, T, state, direction_count)
               if a != policy[index]:
                    policy[index] = a
     display(policy, shape=(1,state_count))

print("=================== FINAL RESULT ==================")
print("Iterations: " + str(iter_count))
print("Delta: " + str(delta))
print("Gamma: " + str(gamma))
print("Epsilon: " + str(epsilon))
print("===================================================")
for j in range(state_count):
     print "State #", j, ": ", utility[j]
print("===================================================")
display(policy, shape=(1,state_count))
print("===================================================")

