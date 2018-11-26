import numpy as np
import copy

### PROBLEM DEFINITION ###
# Attributes used by Learner
state_count = 58
actions = [0, 1, 2, 3, 4]  ## 0 := NORTH || 1 := EAST || 2 := SOUTH || 3 := WEST || 4 := DOWN
direction_count = len(actions)
gamma = .92  ## Future Utilization Discount
epsilon_init = 0.90  ## Initial Exploration vs. Exploitation Tradeoff Variable  (1 = Full Exploration)
epsilon_decay = 0.90 ## Exploration vs. Exploitation Transition Decay (1 = No Change)
episode_count = 0
episode_target = 100
max_moves = 200

# Attributes hidden from Learner
rewards = np.array([
     0, 0, -2, -1, 0, 0, -1, -1, -0.5, 0, \
     -2, -1, -1, -2, -1, -1, -1, -0.5, -1, -1, \
     -1, -0.5, 0, -1, -0.5, -0.5, -0.5, 0, 0, -0.5, \
     -1, -1, 0, 0, -0.5, -1, -0.5, -1, -2, -1, \
     -1, -1.5, -0.5, -1, -1, -1.5, -1.5, 0, \
     0, 0, -1, -1, -1, 0, -0.5, -2, -10, +100])

T = np.load("T_large.npy")

d = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST", 4: "DOWN"}
d2 = {-1:"  %  ", 0:"  N  ", 1:"  E  ", 2: "  S  ", 3: "  W  ", 4: "  D  "}

### LEARNER ###
# Value-function based Q-learning

# Learner Functions
def move(state, recommended_action, transition_matrix, actions):
     while True:
          actions_sans = copy.copy(actions)
          T_vector = transition_matrix[state,:,recommended_action]
          action_percentage = max(T_vector)
          intended_state = -1
          for state_index in range(len(T_vector)):
               if T_vector[state_index] == action_percentage:
                    intended_state = state_index
                    break
          action_draw = np.random.uniform()
          if action_draw <= action_percentage:
               return (intended_state, recommended_action)
          else:
               actions_sans.remove(recommended_action)
               recommended_action = np.random.choice(actions_sans)

# Update Q-Util with new values, weighted by Learning Rate, within episodes
def episodeUtil(move_list, gamma, q_util):
     for index in range(len(move_list)):
          new_util = move_list[index][2]
          for jndex in range(index+1, len(move_list)):
               alpha = 1.0/(jndex+1.0)  ### Learning Rate Function
               new_util += alpha*gamma*move_list[jndex][2]
          move_list[index][2] = new_util
     for move_item in move_list:
          q_util[move_item[0],move_item[1]] = move_item[2]
     return q_util

# Update Q-Util with Utility values from last episode, but only if they're greater
def updateUtil(primary_util, episode_util, state_count, action_count):
     for i in range(state_count):
          for j in range(action_count):
               if episode_util[i,j] > primary_util[i,j] or primary_util[i,j] is None:
                    primary_util[i,j] = episode_util[i,j]
     return primary_util

# Update Q-Map with new values from Q-Util
def updateMap(state_count, q_map, q_util):
     for state in range(state_count):
          if q_map[state] != -1 and not all(entry is None for entry in q_util[state]):
               q_map[state] = np.argmax(q_util[state])
     return q_map

# Initialization
#np.random.seed(10107)
q_map_init = []
for state in range(state_count):
     if state == 0 or state == 57:
          q_map_init.append(-1)
     else:
          q_map_init.append(np.random.choice(actions))
q_map = np.array(q_map_init)
#print "Q-Policy:\n", q_map

q_util = np.array([[None]*direction_count]*state_count)
q_episode = q_util.copy()
#print q_util

#print "Simulation Start!"
for _ in range(episode_target):
     epsilon = epsilon_init*epsilon_decay**(episode_count)
     episode_count += 1
#     print "\nStarting Episode %03d" % (episode_count,)
     move_count = 0
#     print "Epsilon is now", epsilon
     episode_complete = False
     target_reached = "NO"
     current_state = 1
     move_list = []
     while not episode_complete:
          recommended_action = q_map[current_state]  # @01 CHECK FOR BEST ACTION
#          print "Suggested action is", d[recommended_action]
          if np.random.uniform() < epsilon:  # @02 ATTEMPT TO TAKE RECOMMENDED ACTION
               actions_sans = copy.copy(actions)
               actions_sans.remove(recommended_action)
               recommended_action = np.random.choice(actions_sans)
#               print "... However, Agent has prioritized EXPLORATION over EXPLOITATION by moving", d[recommended_action]
#          else:
#               print "Agent has taken suggested action."
          state, action = move(current_state, recommended_action, T, actions)  # @03 GET NEW STATE AND UPDATE VALUES
#          if action != recommended_action:
#               print "Agent has made a mistake!"
          move_count += 1
#          print "For Move #%d, Agent has moved from State %02d to State %02d" % (move_count, current_state, state)
          reward = rewards[state]
          move_list.append([current_state, action, reward, state])
          if q_map[state] == -1:  # @05 CHECK IF EPISODE IS OVER
               episode_complete = True
               if state == 57:
                    target_reached = "YES"
#               print "This episode has come to an end due to State %02d being an absorption state" % (state)
          elif move_count == max_moves:
               episode_complete = True
               if state == 57:
                    target_reached = "YES"
#               print "This episode has come to an end due to reaching %d moves" % max_moves
          else:  # @06 IF NOT, REPEAT
               current_state = state
     print "Move List:\n", move_list
     q_episode = episodeUtil(move_list, gamma, q_episode)   # @07 IF SO, DOCUMENT EPISODE AND START OVER
#     print "Q-Episode has been updated to:\n", q_episode
     q_util = updateUtil(q_util, q_episode, state_count, direction_count)
#     print "Q-Util has been updated to:\n", q_util
#     print "Q-Map was set to:\n", q_map
     q_map = updateMap(state_count, q_map, q_util)
#     print "Q-Map has been updated to:\n", q_map
     policy = ""
     for state_index in q_map:
          policy += str(d2[state_index])
     print "New Policy is:\n", policy
     print "\n###########################################"
     print "##\t\tEpisode %03d\t\t##" % (episode_count,)
     print "###########################################"
     print "##\tMoves Made: %d\t\t\t##" % (move_count)
     print "##\tTotal Rewards Collected: %d\t\t##" % (sum([rewards_counter[2] for rewards_counter in move_list]))
     print "##\tTarget State Reached: %s\t\t##" % (target_reached)
     print "###########################################"

