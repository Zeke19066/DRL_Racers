from copy import copy
import os
import time
from datetime import datetime
import numpy as np
from collections import deque
import json
import pickle

from decorators import function_timer
from Racers_Agent import Agent
import Racers_Wrapper

'''
Notes:

Change the Multiprocess pipe terminal signal from -99999 to "END_TRANSMISSION"

ideas: trigger game over at net zero reward.



Need to breakout reset and meta (time, lap, place, etc) into seperate functions.
Add more plots with button to GUI.
breakout parameters and counters into seperate dicts.
Save permanent variables to JSON: Reward Que, Highest score
only save if score is within n% of highest score; to prevent outliers with glitch.

A target game-time is 1:50 (110 seconds) for a human:
    - Less means that the agent did not finish.
    - More means that an agent is slower than optimal.

Reward/Second is used to track outliers and prevent them from saving by tracking reward/time:
deviance = (current_total_reward/current_session_time)-(avg_total_reward/avg_total_reward)
once this figure exceeds a limit, maybe positive 5reward/second above average, we exclude the save from 
executing once the race is over. A deviant strategy, like wedging in a reward-positive corner, will yield
much higher reward/time, which we filter out.

When the agent is learning from a dataset, it can begin with a 100% fail rate and quickly get its initial bearings, 
usually with forward as the first move it learns. This is because action choices do not affect later outcomes when using 
an on-the-rails Supervised environment. An Unsupervised agent cannot succeed under the same conditions, or else it may 
not encounter later positive rewards once it enters a low-reward state.

Saving Conditions: Two Pools are operating (pool_a, pool_b). Pool_a always learns and saves. This is the main population,
and will be unstable, but perhaps will higher highs with the lows. If a candidate completes the full race, the unedited
state that yielded the run will be saved in pool_b; the winner's circle. Once the circle reaches 5 members, they are averaged
and the pool_a candidate is overwriten with their offspring. Pool_b is then emptied.

Learning Rates:
    1e-6 (0.000001) Classic Rate
    1e-7 (0.0000001) Slow Rate
'''

class ModelTrain():

    def __init__(self, mode):
        print('Initializing Training Model')
        
        self.params = { ## User Set Training Hyperperameters & Game Parameters
            ##Hyperparameters
            'actor_learning_rate':7e-7, #5e-6, 9e-6, 1e-6, 5e-4
            'critic_learning_rate':7e-7, #5e-6, 9e-6, 1e-6, 5e-4
            'gamma': 0.995, #0.95; Discount Factor
            'tau': 0.98, #0.8, 0.65; Scaling parameter for GAE aka lambda
            'beta': 2, #0.01; Entropy coefficient; induces more random exploration
            'epsilon': 0.1, #parameter for Clipped Surrogate Objective
            'epochs': 2, #5unsup, 1supervised;how many times do you want to learn from the data
            'cycle_limit': 20000000,
            'mini_batch_size': 200, #5; Used for dataset learning.
            'num_local_steps': 800, #20; Used for dataset learning. total learning steps at learn time.]
            'merge_pop': 5, #how many parents per merge.
            'fps_target':6, #we slow down to achieve this rate. Max 30

            ##Game Parameters
            'min_reward':-5, 'max_reward':5, #0.5; what the rewards are scaled to
            'reward_rate_limit': 20, #5; see notes above. Threshold that will triger save-skip.
            'reward_mode':0, #0 is minmap, 1 is speedguage
            'negative_reward_limit': 500, #sum TOTAL negative rewards before reset?
            'lap_time_limit': 120, #90; max time per lap before we time-out reset.
            'fail_time_limit': 3, #10; how many seconds of consecutive negative rewards before reset?
            'reward_ratio_limit': 0.95, #0.75; limit of % wrong answers.
            'stuck_limit': 250, #sum negative rewards consecutive before reset?
            'off_track_start_1':4, #time in seconds
            #bonus based on position; below place values are
            #multuplied by the max reward and then added as a bonus
            'place_bonus':{1:0.4, 2:0.2, 3:0.16, 4:0.12, 5:0.08, 6:0},
            'hard_reset': 15000, #hard reset every n cycles.
            }

        self.counters = { ## Initialize counters & ques.
            'final_accuracy':0,
            'iteration': 1,
            'game_count':1,
            'feed_reward_que':deque(maxlen=round(0.5 #feed an avg to the agent
                                    *self.params['fps_target'])),
            'side_glitch_que':deque(maxlen=round(1.6
                                    *self.params['fps_target'])),
            'lap':1,
            'lap_time':0,
            'lap_time_dict':{1:0, 2:0, 3:0}, #stores the laptimes
            'game_time':1, #tally of play time in seconds. Menu/reset does not count.
            'hist_time':1, #time from prior sessions
            'total_time':1, #real training time aggregate
            'stuck_counter': 0, #Counter for consecutive negative rewards.
            'last_place_counter':0,#count consecutive cycles in last place
            'final_reward_score':0, #total reward at last reset.
            #each que contains [miss count, total count] for each game
            'session_move_accuracy': [deque([[0,1]],maxlen=25), 
                                      deque([[0,1]],maxlen=25), deque([[0,1]],maxlen=25),
                                      deque([[0,1]],maxlen=25), deque([[0,1]],maxlen=25)], 
            #[miss count, total count] for current game. Aggregated w/ above^
            'local_move_accuracy':[[0,1], [0,1], [0,1], [0,1], [0,1]], 
            
            'reward_que':deque(maxlen=25),
            'reward_que_avg': 1,
            'race_time_que':deque(maxlen=25), #Both finish and non-finish races
            'race_time_que_avg': 1,
            'finish_time_que':deque(maxlen=25), #Only counts time of races fully completed (3 laps)
            'finish_time_que_avg': 0,
            'accuracy_que':deque(maxlen=25),
            'accuracy_que_avg': 1,
            'agent_qmax_list': [],
            'agent_qmax_avg':0,
            'critic_qmax_list': [],
            'critic_qmax_avg':0,
            'reward_rate': [0, 0], #[reward/time, rewardavg/timeavg]
            'plus_sum': 0, #track the sum of positive rewards for the race.
            'plus_que':deque(maxlen=25), #get an avg from the que
            'plus_avg':0,
            
            #supervised counters
            'session_move_dist':deque([[0,0,0,0,0]], maxlen=25), #move dist for the last 25 games.
            'local_move_dist':[0,0,0,0,0], #move dist for this race.

            'completion_que':deque(maxlen=25), #
            'completion_count':0, #how many races completed all laps.
            'reward_polarity': [0,0],#positive count, negative count of rewards.
            'repeat_counter': [0,0], #[current repeat, repeat limit]
            'quit_reset_bool': False,
            'race_over_bool': False,
            'reset_bool':False,
            'parent_counter':1,
            'time_offset':0.1, #time pause to achieve desired framerate.
            'fail_streak_triple':[0,0,0], #current, session, overall
            'win_count': 0, 'loss_count':0, 'negative_reward_count': 0, #loss is permanent like win, fail is per 'session' and is reset
            }

        #for individual entries
        self.benchmarks = {
            'total_reward_trend':0,
            'reward_rate':0,
            'race_time_trend':0,
            'completion_rate':0,
            }

        ##Load Counters & Benchmarks JSON
        self.benchmarks_path = r'pretrained_model\benchmarks.json'
        self.counters_path = r'pretrained_model\counters.pkl'
        self.home_dir = os.getcwd()
        self.logger(mode='load')

        #the aggregate of all entries.
        self.benchmarks_history = {'training_cycles':0,
                                    'training_time':0,
                                    'game_count':0,
                                    'completion_count':0,
                                    }

        self.accuracy = 0
        self.first_run = True
        self.main_mode = mode
        self.cycles_per_second = "Initializing"
        self.file_name = ""
        self.total_move_accuracy = [[0,0], [0,0], [0,0], 
                                    [0,0], [0,0]]
        self.total_move_dist = [0,0,0,0,0]

        print('... launching Agent', end="")
        self.agent = Agent(self.params, num_actions=5)
        print('... launching Screen', end="")
        self.screen = Racers_Wrapper.ScreenGrab(dataset_mode="Solo")
        print('... launching Controller',end="")
        self.controller = Racers_Wrapper.Actions()
        print('... Initialization Complete.')

    #@function_timer
    def train(self, start, send_connection):

        self.counters['reset_bool'] = False
        self.counters['race_over_bool'] = False
        self.total_reward, self.subcycle = 1, 0
        self.last_reset = self.counters['iteration']
        self.shot_counter, reward_scan = 0, 0
        self.race_time_delta = 0
        self.fail_time_start, self.fail_time_bool = 0, False
        self.side_glitch_state_bool = False 
        self.completion_bool = False
        self.controller.current_mapmode = self.params['reward_mode']

        image_data = self.screen.quick_Grab()
        image_data = self.screen.add_watermark(image_data, 0) #0 is forward
        image_data = self.screen.resize(image_data, False)
        image_tensor = self.agent.image_to_tensor(image_data)

        if (self.main_mode != "train_supervised"
            and self.first_run):
            self.controller.first_Run()
            self.first_run = False

        self.agent.initial_state_bool = True
        self.controller.Reset() # Reset the game state for a clean start.
        self.subtime = datetime.now()
        image_data = self.screen.quick_Grab()
        self.controller.toggle_pause(True)
        image_data = self.screen.add_watermark(image_data, 0) #0 is forward
        image_data = self.screen.resize(image_data, False)
        image_tensor = self.agent.image_to_tensor(image_data)
        _, _, _, _ = self.agent.choose_action(image_tensor)
        self.controller.action_Map(0)

        self.controller.toggle_pause(False)

        ## Main cycle loop
        print('Entering Training Loop')
        while (self.counters['iteration'] < self.params['cycle_limit']):
            
            self.subcycle = max(self.counters['iteration'] 
                                - self.last_reset, 1)
            #Get NN output
            action, prob, crit_val, action_space = self.agent.choose_action(image_tensor)

            ## Get qmax with decimal accuracy
            action_space*= 10 #fast decimal, allows int detachment with accuracy.
            action_space = action_space.detach().cpu().numpy().astype('int32') #its faster if we turn it into ints.float32, int32
            action_space = action_space/10#now get it back to the right decimal place.
            action_space = action_space[0]
            qmax = np.max(action_space)

            self.np_action = action

            ## Execute action
            if ((self.race_time_delta < self.params['off_track_start_1']) 
                    and (self.np_action != 0)): #straight at start
                self.controller.action_Map(0) # Override Action
            elif self.np_action == 0:
                 #1/100 chance to fire powerup.
                powerup_rand = np.random.randint(99)
                if not powerup_rand: #on zero
                    self.controller.action_Map(1) #Override Action
                else:
                    self.controller.action_Map(self.np_action) #Agent Action
            else: #Normal Condition
                self.controller.action_Map(self.np_action) #Agent Action
            

            #slow down to achieve desired framerate.
            time.sleep(self.counters['time_offset'])

            self.image_data_ = self.screen.quick_Grab() #Get the next screenshot
            self.image_data_ = self.screen.add_watermark(self.image_data_,
                                                        self.np_action)

            ## Get meta
            reward_scan, wrong_way_bool = self.screen.reward_scan(
                                                    self.image_data_,
                                                    self.params['reward_mode'],
                                                    agent_action=self.np_action)
    
            ## Local reward rules
            if self.params['reward_mode'] == 0: #Minmap
                sub_reward, terminal = self.local_rules_minmap(reward_scan, 
                                                              wrong_way_bool)
            
            elif self.params['reward_mode'] == 1:#Speed
               sub_reward, terminal = self.local_rules_speed(reward_scan, 
                                                            wrong_way_bool)

            self.image_data_ = self.screen.resize(self.image_data_,
                                                 wrong_way_bool)
            image_tensor_ = self.agent.image_to_tensor(self.image_data_)
            self.agent.remember(image_tensor, action, prob,
                                crit_val, sub_reward, terminal)
            
            image_tensor = image_tensor_
            self.counters['iteration'] += 1

            ## Quit-reset the game once per n cycles to limit glitches
            if self.counters['iteration'] % self.params['hard_reset'] == 0:
                self.counters['quit_reset_bool'] = True

            ## Multiprocessing pipe to send metrics to GUI
            gui_deviance = (self.counters['reward_rate']
                            +[self.params['reward_rate_limit']])
            self.counters['agent_qmax_list'].append(qmax)
            val = round(np.mean(self.counters['agent_qmax_list']),1)
            self.counters['agent_qmax_avg'] = val
            self.counters['critic_qmax_list'].append(crit_val)
            val = round(np.mean(self.counters['critic_qmax_list']),1)
            self.counters['critic_qmax_avg'] = val
            lap_metrics = [self.counters['lap'],self.counters['lap_time']]
            action = np.argmax(action_space)
            hyperparam_container = [self.params['actor_learning_rate'],
                                    self.params['mini_batch_size'],
                                    self.params['epochs']
                                    ]

            subtime = int((datetime.now() - start).total_seconds())
            total_time = subtime + self.counters['hist_time']
            self.counters['total_time'] = total_time
            
            self.race_time_delta = (datetime.now()
                                    - self.subtime).total_seconds()

                                                                            ###
            try:
                metrics = [ #Game Metrics
                    action, sub_reward,                                #0,1
                    [action_space],self.total_reward,                  #2,3            
                    self.subcycle, self.agent.loss_print,              #4,5
                    self.counters['iteration'],                        #6
                    self.counters['total_time'], qmax,                 #7,8
                    self.counters['reward_polarity'],                  #9
                    self.counters['race_time_que_avg'], int(crit_val), #10,11
                    self.cycles_per_second,                            #12
                    self.counters['agent_qmax_avg'],                   #13
                    
                    hyperparam_container,                              #14
                    self.counters['game_time'],                        #15
                    self.counters['critic_qmax_avg'],                  #16
                    self.race_time_delta,                              #17
                    9999999999999,                                     #18
                    self.counters['finish_time_que_avg'],              #19
                    self.counters['game_count'],                       #20
                    self.counters['reward_que_avg'],                   #21
                    self.completion_bool, lap_metrics, gui_deviance,   #22-24
                    self.total_move_accuracy                           #25
                ]

                self.gui_send(send_connection, metrics)
            except Exception as e:
                #below string means GUI wasnt launched
                if str(e) != "'bool' object has no attribute 'send'":
                    print("Send Failure:",e)

            ## Trigger reset; GAME OVER
            if self.counters['reset_bool']:
                self.controller.toggle_pause(True)
                self.training_reset()#learning happens here
                time.sleep(2)
                self.controller.toggle_pause(False)
                return #this will kick us back into the main function below and restart training.

    def train_supervised(self, start, send_connection):
        #store tracking variables.
        self.total_reward = 1
        self.subcycle = 0
        self.last_reset = self.counters['iteration']
        self.last_learn = 0
        reward_scan, self.shot_counter = 0, 0
        self.counters['reward_polarity'] = [0,0]

        self.agent.initial_state_bool = True
        self.counters['fail_streak_triple'][1] = 0 #reset the session failstreak

        #Chose the random folder and load the first image.
        image_data, self.human_action, self.counters['race_over_bool'] = self.screen.data_loader(first_run=True)

        #set fail limit to half len.
        self.params['negative_reward_limit'] = round(self.screen.data_len/2)

        image_data = self.screen.add_watermark_dataset(image_data,0) #0 is forward
        image_tensor = self.agent.image_to_tensor(image_data)
        self.image_data_ = image_data
        self.subtime = datetime.now()
        dataset_path = "Dataset//Solo//" + self.screen.subfolder

        # main cycle loop
        print('Entering Training Loop')
        while self.counters['iteration'] < self.params['cycle_limit']: 
            self.counters['reset_bool'] = False #Triggers Reset
            self.counters['race_over_bool'] = False #Completed Race
            terminal = False #Final State
            self.subcycle = max(self.counters['iteration'] - self.last_reset, 1)
            self.action, prob, crit_val, action_space = self.agent.choose_action(image_tensor)

            self.np_action = self.action
            reward, repeat_bool, terminal = self.local_rules_dataset(self.np_action, self.human_action) # Get Reward
            if not repeat_bool: #only update if we're not repeating
                self.image_data_, self.human_action, self.counters['race_over_bool'] = self.screen.data_loader(first_run=False)
                self.image_data_ = self.screen.add_watermark_dataset(self.image_data_,self.np_action) #0 is forward

            # Failed or finished
            if terminal or self.counters['race_over_bool']:
                terminal = True
                self.counters['race_over_bool'] = True
                self.counters['reset_bool'] = True

            image_tensor_ = self.agent.image_to_tensor(self.image_data_)
            terminal = self.counters['race_over_bool']
            self.agent.remember(image_tensor, self.action, prob, crit_val, reward, terminal)
            
            #""" A2C Learning structure
            if (self.screen.img_index - self.last_learn > 500
                 and self.completion_percent % 25 == 0):
                self.last_learn = self.screen.img_index
                #print("Learning......", end="")
                self.agent.learn()
                #print("Done!")
            #"""

            image_tensor = image_tensor_
            self.counters['iteration'] += 1

            #Multiprocessing pipe to send metrics to GUI
            ## Get qmax with decimal accuracy
            action_space*= 10 #fast decimal, allows int detachment with accuracy.
            action_space = action_space.detach().cpu().numpy().astype('int32') #its faster if we turn it into ints.float32, int32
            action_space = action_space/10#now get it back to the right decimal place.
            action_space = action_space[0]
            qmax = np.max(action_space)
            value = int(crit_val)
            
            if self.subcycle % 5 == 0:
                subtime = int((datetime.now() - start).total_seconds())
                total_time = subtime + self.counters['hist_time']
                self.counters['total_time'] = total_time

            var = self.counters['reward_polarity'][0]/self.screen.img_index
            self.accuracy = round(var*100,2) #subcycle would include failed repeats in the accuracy.
            self.completion_percent = round(self.screen.img_index/len(self.screen.action_list)*100)
            frame_double = [self.screen.img_index,self.completion_percent]
            image_double = [self.screen.subfolder, self.screen.img_index] #directory, frame#
            self.counters['agent_qmax_list'].append(qmax)
            self.counters['agent_qmax_avg'] = round(np.mean(self.counters['agent_qmax_list']),1)
            self.counters['critic_qmax_list'].append(crit_val)
            self.counters['critic_qmax_avg'] = round(np.mean(self.counters['critic_qmax_list']),1)
            
            #Terminal Printout
            #if self.counters['iteration'] % 100 == 0:
            #    print(f"{self.counters['iteration']} Subcycle:{self.subcycle}/{completion_percent}% W/L:{self.accuracy}%")

            try:
                metrics = [ #Game Metrics
                    self.np_action, reward, [action_space],     #0-2
                    self.accuracy, self.subcycle,               #3,4
                    self.counters['race_time_que_avg'],         #5
                    self.counters['iteration'],                 #6
                    self.counters['total_time'], qmax,          #7,8
                    self.params['mini_batch_size'],             #9
                    self.counters['accuracy_que_avg'],          #10
                    value, self.cycles_per_second,              #11,12
                    self.counters['agent_qmax_avg'],            #13
                    self.params['actor_learning_rate'],         #14
                    self.counters['game_time'],                 #15
                    self.counters['final_accuracy'],            #16
                    self.counters['fail_streak_triple'],        #17
                    self.params['mini_batch_size'],             #18
                    self.counters['finish_time_que_avg'],       #19
                    self.counters['game_count'],                #20
                    self.total_move_accuracy,                   #21
                    dataset_path, frame_double,                 #22,23
                    self.counters['critic_qmax_avg'],           #24
                    self.total_move_dist                        #25
                    ]

                self.gui_send(send_connection, metrics)
            except Exception as e: 
                print("Send Failure:",e)

            # Trigger reset; GAME OVER
            if self.counters['reset_bool']:
                time_delta = (datetime.now() - self.subtime).total_seconds()
                self.counters['game_time'] += int(time_delta)
                self.cycles_per_second = round(self.subcycle/time_delta,2)

                #handle the session_move_accuracy:
                for i in range(5): #For current game
                    double = self.counters['local_move_accuracy'][i]
                    self.counters['session_move_accuracy'][i].append(double)
                self.counters['local_move_accuracy'] = [[0,1], [0,1], [0,1], [0,1], [0,1]]

                #handle the move-dist:
                dist = self.counters['local_move_dist']
                self.counters['session_move_dist'].append(dist)
                self.counters['local_move_dist'] = [0,0,0,0,0]

                #self.total_reward = max(self.total_reward,10) #set a floor of 10.
                self.counters['game_count'] += 1
                #self.counters['accuracy_que'].append(self.total_reward) #let's track how we did.
                self.counters['accuracy_que'].append(self.accuracy)
                self.counters['accuracy_que_avg'] = round(np.mean(self.counters['accuracy_que']),2)
                self.counters['agent_qmax_list'], self.counters['critic_qmax_list'] = [], []
                #Terminal Printout GameOver
                var = (self.counters['reward_polarity'][0]/self.screen.img_index)
                self.counters['final_accuracy'] = round(var*100,2)
                print(f"Game Over!         Final Accuracy:{self.counters['final_accuracy']}%         25 Game Avg:{self.counters['accuracy_que_avg']}%")

                self.logger(mode='save', benchmarks=False) #log before var reset below.

                """
                print("Learning......", end="")
                self.agent.learn() #we learn once the race is over.
                print("Done!")
                """

                self.counters['final_reward_score'] = self.total_reward
                self.counters['agent_qmax_list'], self.counters['critic_qmax_list'] = [], []
                self.counters['reward_polarity'] = [0,0]
                self.total_reward = 0
                self.counters['back_on_track_bonus'] = 0
                self.counters['negative_reward_count'] = 0
                self.last_reset = self.counters['iteration']

                self.subtime = datetime.now()
                self.agent.save_models()
                #self.controller.toggle_pause(True)
                return #this will kick us back into the main function below and restart training.

    # Testing Environment
    def test(self, start, send_connection):
        pass
    
    # Reset environment; Start a fresh race.
    def training_reset(self):
        self.counters['game_time'] += int(self.race_time_delta)
        self.cycles_per_second = round(self.subcycle/self.race_time_delta,2)

        ## Call Learning
        self.local_learn()

        if self.counters['quit_reset_bool']:
            self.counters['quit_reset_bool'] = False
            self.controller.quit_Reset()

        #calculate some counters
        self.counters['reward_que'].append(self.total_reward)
        val = round(np.mean(self.counters['reward_que']),2)
        self.counters['reward_que_avg'] = val
        self.counters['race_time_que'].append(self.race_time_delta)
        val = round(np.mean(self.counters['race_time_que']),1)
        self.counters['race_time_que_avg'] = val
        self.counters['plus_que'].append(self.counters['plus_sum'])
        self.counters['plus_avg'] = np.mean(self.counters['plus_que'])
        self.counters['completion_que'].append(0)#assume we didnt finish
        self.counters['game_count'] += 1
        print(f"Time Offset: {self.counters['time_offset']}")
        if self.cycles_per_second >= self.params['fps_target']:
            self.counters['time_offset'] += 0.01
        elif self.cycles_per_second <= self.params['fps_target']:
            self.counters['time_offset'] -= 0.01
            #set a floor; cant wait negative time
            self.counters['time_offset'] = max(self.counters['time_offset'],0.01)

        if self.completion_bool: #we finished all laps
            self.counters['completion_count'] += 1
            self.counters['completion_que'].pop()
            self.counters['completion_que'].append(1)
            self.counters['finish_time_que'].append(self.race_time_delta)
            val = round(np.mean(self.counters['finish_time_que']),1)
            self.counters['finish_time_que_avg'] = val
        
        self.logger(mode='save') #log before var reset below.

        for i in range(5):#handle the session_move_accuracy:
            double = self.counters['local_move_accuracy'][i]
            self.counters['session_move_accuracy'][i].append(double)
        
        
        #reset some counters
        self.counters['local_move_accuracy'] = {0:[0,1], 1:[0,1], 2:[0,1],
                                                3:[0,1], 4:[0,1]}
        self.counters['agent_qmax_list'] = []
        self.counters['critic_qmax_list'] = []
        self.counters['reward_polarity']=[0,0]
        self.counters['lap'] = 1
        self.counters['lap_time'] = 0
        self.counters['lap_time_dict'] = {1:0, 2:0, 3:0}
        self.counters['plus_sum'] = 0

        self.counters['final_reward_score'] = self.total_reward
        self.counters['back_on_track_bonus'] = 0
        self.counters['negative_reward_count'] = 0
        self.last_reset = self.counters['iteration']
        self.subtime = datetime.now()

        return

    def local_learn(self):
        """
        (condition_1)
        To begin, only racetime > average is required to become a parent.

        Once we've finished (all laps) for more than 10 races, then
        racetime < average and finishing is required to become a parent.
        """
        condition_1 = True
        #have we scored above avg positive rewards? (not net)
        #if self.counters['plus_sum'] > self.counters['plus_avg']:
        #    condition_1 = True

        #Determine if instance shall become a parent
        parent_bool = False
        """This handles the merge protocol.
        if (self.counters['completion_count'] < 10 
            and condition_1:
            parent_bool = True
        elif (self.counters['completion_count'] >= 10 
             and self.completion_bool 
             and self.race_time_delta < self.counters['finish_time_que_avg']):
            parent_bool = True
        """

        #Save parent
        if parent_bool:
            #save the parent for a later merge.
            label = self.counters['parent_counter']
            merge_name = f"pretrained_model/Merge_Folder/parent{label}.pth"
            #only the actor is merged.
            self.agent.actor_model.save_checkpoint(file_name=merge_name)
            print(f"Saving Parent:{label}")
            self.counters['parent_counter'] += 1

        #Learn if we're not merging.
        if ((self.counters['parent_counter'] < (self.params['merge_pop']+1))
            and condition_1):
            print("Learning......", end="")
            self.agent.learn() #we learn once the race is over.
            print("Done!")
        else:#don't learn, a merge is due
            self.agent.memory.clear_memory()

        #Merge
        if self.counters['parent_counter'] == (self.params['merge_pop']+1): #time to merge
            self.counters['parent_counter'] = 1 #reset the counter
            self.agent.merge_models()
            #saving here would overwrite the merge^
        else:
            self.agent.save_models()
        
    ##Rules Block
    def local_rules_minmap(self,reward_scan, wrong_way_bool):
        #Rewerds occur in 2/-2 range and are scaled to the max/min range.
        terminal, endgame_attempt, side_glitch_bool = False, False, False
        self.total_move_accuracy = [[0,0], [0,0], [0,0], 
                                    [0,0], [0,0]]
        side_set = (2,3,4) #left, right, reverse
        reward, place = reward_scan, 6
        lap_bonus_bool, lap_reward = False, 1

        #combine the aggreagate with the current game.
        for i in range(5): #it's a 5 item number-key dict. For agg
            temp_que = list(self.counters['session_move_accuracy'][i])
            for double in temp_que:
                self.total_move_accuracy[i][0] += double[0]
                self.total_move_accuracy[i][1] += double[1]
        for i in range(5): #For current game
            double = self.counters['local_move_accuracy'][i]
            self.total_move_accuracy[i][0] += double[0]
            self.total_move_accuracy[i][1] += double[1]
        
        #update the local move counter; move:[miss count, total count]
        self.counters['local_move_accuracy'][self.np_action][1] += 1 

        ##Determine our reward rate[reward/time, rewardavg/timeavg]
        if self.subcycle>10:
            reward_rate = round(self.total_reward/self.race_time_delta,2)
            reward_rate_avg = round(self.counters['reward_que_avg']
                                    /self.counters['race_time_que_avg'],2)
            self.counters['reward_rate'][0] = reward_rate
            self.counters['reward_rate'][1] = reward_rate_avg

        ##Determine our lap Time:
        if self.counters['lap'] == 1:
            self.counters['lap_time'] = self.race_time_delta
        elif self.counters['lap'] != 1:
            subtime = self.counters['lap_time_dict'][self.counters['lap']-1]
            self.counters['lap_time'] = self.race_time_delta - subtime

        ##Determine what lap we're on
        if self.counters['lap_time'] > 25:
            lap_img = self.image_data_[21:72, 632:706]
            lap_check = self.screen.lap_scan(lap_img)
            if lap_check != self.counters['lap']: #we are on the next lap.
                self.counters['lap_time_dict'][self.counters['lap']] = self.race_time_delta
                self.counters['lap'] = lap_check
                lap_bonus_bool = True

        ##See if we've Timed-Out(we get n seconds per lap or reset occurs)
        if self.counters['lap_time'] > self.params['lap_time_limit']:
                endgame_attempt = True
                print("TIMED OUT.......Resetting")

        ##See if we've Finished
        if self.race_time_delta > 100: #check to see if we've finsihed the race.
            finish_image = self.image_data_[165:252, 340:528]
            self.counters['race_over_bool'] = self.screen.race_over(finish_image)
            if self.counters['race_over_bool']:
                self.completion_bool = True
                reward = 3
                endgame_attempt = True

        ##Check for side glitch
        #too many hits triggers a state of minimum reward, until 3 seconds without sideglitch
        if self.subcycle % 2 == 0 and reward >= 0:#for efficiency, check only sparingly
            side_glitch_bool = self.screen.side_glitch(self.image_data_)
            if side_glitch_bool:
                wrong_way_bool = True
                self.counters['side_glitch_que'].append(1)
                reward = -2
                val_1 = sum(self.counters['side_glitch_que'])
                val_2 = len(self.counters['side_glitch_que'])
                side_glitch_percent = round(val_1/val_2*100,1)
                if side_glitch_percent >= 25: #Enter a sideglitch state
                    endgame_attempt = True
                    self.side_glitch_state_bool = True
                    print(f'SIDE GLITCH THRESHOLD!')
                else:
                    self.side_glitch_state_bool = False
            if not side_glitch_bool:
                self.counters['side_glitch_que'].append(0)
        if self.side_glitch_state_bool == True:
            reward = -2

        ##Reverse 
        if self.np_action == 2:
            reward = -1.5

        ##Power Up
        if self.np_action == 1:
            powerup_bool = self.screen.item_scan(self.image_data_)
            if powerup_bool: 
                reward += 0.5

        if wrong_way_bool:
                reward = -2

        ##No turns at start.
        if self.race_time_delta < self.params['off_track_start_1']:  
            if self.np_action in side_set:
                reward = -2
                #endgame_attempt = True

        ##this is for the stuck time & failrate
        if reward <= 0: #Negative Reward
            #[positive count, negative count]
            self.counters['reward_polarity'][1] += 1
            self.counters['local_move_accuracy'][self.np_action][0] +=1 

            #Reset if we've been stuck too long
            if not self.fail_time_bool:
                self.fail_time_bool = True
                self.fail_time_start = self.race_time_delta
            elif self.fail_time_bool:
                    subtime = self.race_time_delta - self.fail_time_start
                    if subtime > self.params['fail_time_limit']:
                        self.fail_time_bool = False
                        endgame_attempt = True
                        print("Reset: Stuck too long ",end="")
                        print(self.params['fail_time_limit'])

            #if we exceed the failrate
            failrate = (self.counters['reward_polarity'][1]
                        /sum(self.counters['reward_polarity']))
            if failrate >= self.params['reward_ratio_limit']:
                reward = -2
                endgame_attempt = True

            #Consecutive failures lower the reward incrementally.
            fail_time = self.race_time_delta-self.fail_time_start
            fail_time_reward = (fail_time/15)*1
            reward -= fail_time_reward

        elif reward > 0: #Positive Reward (place-check)
            self.fail_time_bool = False
            self.counters['reward_polarity'][0] += 1 #[positive count, negative count]
            self.counters['plus_sum'] += reward #add the positive reward to the sum
            place = self.screen.place_scan(self.image_data_[87:88, 51:87])

            
        ##prevent early reset.
        if (endgame_attempt 
            and (self.race_time_delta > 10
                 or self.counters['lap']>1)):
            terminal = True
            self.counters['reset_bool'] = True


        ##Normalize reward bounds.
        reward = max(min(reward, 2),-2)
        #like c map function:(input val, [in_min,in_max],[out_min,out_max])
        reward = np.interp(reward,[-2, 2],
                           [self.params['min_reward'],
                            self.params['max_reward']]) 
        self.counters['feed_reward_que'].append(reward)
        avg_reward = round(np.mean(self.counters['feed_reward_que']),2)
        ##Overrides & adjustments.


        #Survival Bonus if we're not finishing races
        if (self.counters['completion_count'] < 10
              and reward > 0):
            time_target = 120 #target time for completing the race.
            s_bonus = round(self.race_time_delta/time_target, 2)
            s_bouns = min(s_bonus,1)
            avg_reward += s_bouns

        #Place Bonus if we're finishing races.
        if (self.counters['completion_count'] >= 10
            and reward > 0):
            bonus = (self.params['max_reward']
                     *self.params['place_bonus'][place])
            avg_reward += bonus
        if lap_bonus_bool:
            avg_reward = lap_reward
        if side_glitch_bool:
            avg_reward = self.params['min_reward']

        ##Final Reward Processing.
        self.total_reward += avg_reward
        self.total_reward = round(self.total_reward, 4)
        return avg_reward, terminal
    
    def local_rules_speed(self, reward_scan, wrong_way_bool, place):
        reward = reward_scan
        terminal = False
        endgame_attempt = False
        maximum, minimum = 2, -2

        if self.counters['race_over_bool']: #We finished the race
            self.completion_bool = True
            endgame_attempt = True
            print("Reset: Race is over!")

        if self.race_time_delta < 3 and self.np_action != 2: #catch those pesky false negative scores at start.
            if reward <= 0:
                reward = 1
        
        #for reverse actions
        if self.np_action == 2:
            reward = minimum
            #self.counters['negative_reward_count'] -= 0.5 #just an offset        
        
        """#stop spamming shot button (no move)
        if self.np_action == 1:
            self.shot_counter +=1
            #reward -= 5
            if self.shot_counter > 10:
                reward = minimum

        elif self.np_action != 1:
            self.shot_counter = 0
        """

        #incetivize going straight at the start; override pesky cannon shot.
        if self.race_time_delta < 1.5:
            reward = max(reward, 0.1) #set a floor
            if self.np_action != 0:
                reward = minimum
            
        #Wrong Way
        if wrong_way_bool:
            reward = minimum

        reward = round(max(min(reward, maximum), minimum),2) #cap the reward.
        self.total_reward += reward
        self.total_reward = round(self.total_reward,2)

        #this is for the stuck counter.
        #In the Speedometer paradigm, too many negatives mean we're stuck.
        if reward <= 0:
            self.counters['negative_reward_count'] += 1 #abs(reward)
            self.counters['stuck_counter'] += 1 #abs(reward)
            #if we hit the stuck limit
            if (self.counters['stuck_counter'] 
                >= self.params['negative_reward_limit']):
                self.counters['stuck_counter'] = 0
                endgame_attempt = True
                #print("Reset: Too many negative rewards")
            
            #if we hit the total failure limit
            if (self.counters['negative_reward_count'] 
                    >= self.params['negative_reward_limit']):
                reward = minimum
                endgame_attempt = True
            
        elif reward > 0:
            self.counters['stuck_counter'] = 0

        #prevent early reset.
        if endgame_attempt and self.race_time_delta > 5:
                terminal = True
                self.counters['reset_bool'] = True
        
        return reward, terminal

    def local_rules_dataset(self, agent_action, human_action):
        """
        Since turning is volitile, we need to over-represent turns and underrepresent going straight.
        We will set a target distribution of the moveset, and use repeats on failed moves to shift the
        agents experience towards the target distribution. To achieve this, we will track the actual
        experienced distribution against the target, and ration repeats towards that goal.
        We will track:
            - Experienced Distribution last 25 races
            - Target Distribution
            The delta % will then be applied to the 'max repeats' to determine the actual
            number of repeats for that instance of repeat.

        Also, the self.counters['session_move_accuracy'] item has to be tracked as a moving sum
            of the last n games, so that it is dynamic through the entire session, 
            but also doesn't have a hard reset to zeros. 
        This will be achieved through a deque containing lists of the previous game.
        The active game is only contributed to the master count once the game is over and the next
        round starts. The current game's set will be added to the master set while current.
        
        self.action_set = [0'Forward', 1'Powerup', 2'Reverse',
                           3'Left', 4'Right']

        target ratio is 20/45/35

        {0: 60.71, 1: 0.0, 2: 0.0, 3: 25.15, 4: 14.14}
        Current Moveset Frequency = {Forward: 60.71%, Powerup: 0.0%, 
                                     Forward: 0.0%, Left: 25.15%,
                                     Right: 14.14%}
        """

        repeat_max = 10 #Maximum number of repeats.
        mistake_tolerance = 0.2 #0.20; what % of moves can we miss before a miss triggers a repeat for that move?
        repeat_bool = False
        target_dist = {0:0.2, 1:0, 2:0, 3:0.45, 4:0.35}

        self.total_move_accuracy = [[0,0], [0,0],
                                    [0,0], [0,0], [0,0]]
        
        self.total_move_dist = [0,0,0,0,0]

        #determine the aggreagate accuracy.
        for i in range(5): #it's a 5 item number-key dict. For agg
            temp_que = list(self.counters['session_move_accuracy'][i])
            for double in temp_que:
                self.total_move_accuracy[i][0] += double[0]
                self.total_move_accuracy[i][1] += double[1]
        for i in range(5): #For current game
            double = self.counters['local_move_accuracy'][i]
            self.total_move_accuracy[i][0] += double[0]
            self.total_move_accuracy[i][1] += double[1]
        
        #determine the aggregate move-dist.
        temp_dist = np.array(self.counters['session_move_dist'])
        temp_dist = temp_dist + np.array(self.counters['local_move_dist'])
        move_dist = temp_dist.transpose() #arrange by index
        total = np.sum(move_dist)
        for i in range(5):
            self.total_move_dist[i] = round(np.sum(move_dist[i])/total,3)


        """ Repeats happen no matter what now.
        #determine how many repeats is limit:
        delta = target_dist[human_action]-self.total_move_dist[human_action]
        delta = max(0,delta)
        #like c map function:(input val, [in_min,in_max],[out_min,out_max])
        limit = np.interp(delta, [0, 0.25], [0, repeat_max]) 

        self.counters['repeat_counter'][1] = limit

        #now handle the repeats
        if self.counters['game_count'] >= 2:
            self.counters['repeat_counter'][0] += 1
            if self.counters['repeat_counter'][0] <= self.counters['repeat_counter'][1]:
                repeat_bool = True
            elif self.counters['repeat_counter'][0] > self.counters['repeat_counter'][1]:
                self.counters['repeat_counter'] = [0,0]
                repeat_bool = False
        """

        #update the local move accuracy; move:[miss count, total count]
        if not repeat_bool: #We're not in repeats
            self.counters['local_move_accuracy'][human_action][1] += 1 
        #update the local move-dist
        self.counters['local_move_dist'][human_action] += 1

        #Agent correct
        if agent_action == human_action:
            self.counters['fail_streak_triple'][0] = 0

            if not repeat_bool: #We're not in repeats
                self.counters['reward_polarity'][0] += 1
            self.counters['repeat_counter'] = [0,0]
            
            current_fail_rate = (self.total_move_accuracy[human_action][0]
                                /self.total_move_accuracy[human_action][1])
            
            #bonus = np.interp(current_fail_rate, [0, 1], [0, 25])
            #reward = 1.75 + bonus
            reward = 2

        #Agent wrong
        elif agent_action != human_action:
            #reward_polarity:[positive count, negative count] of rewards.
            if not repeat_bool: #We're not in repeats
                self.counters['local_move_accuracy'][human_action][0] +=1
                self.counters['reward_polarity'][1] += 1
            
            current_fail_rate = (self.total_move_accuracy[human_action][0]/
                                self.total_move_accuracy[human_action][1])
            

            #bonus = np.interp(current_fail_rate, [0, 1], [0, 25])
            #reward = -1 * (1.75 + bonus)
            reward = -2


        if human_action == 0 and repeat_bool:
            print('SOMETHING IS WRONG')

        #handle the failstreak triple [current, session, overall]
        # (for tracking pusposes)
        self.counters['fail_streak_triple'][0] += 1 #incriment the current failstreak
        if self.counters['fail_streak_triple'][0] > self.counters['fail_streak_triple'][1]: #Current>Sesssion
            self.counters['fail_streak_triple'][1] = self.counters['fail_streak_triple'][0]
            if self.counters['fail_streak_triple'][1] > self.counters['fail_streak_triple'][2]: #Session>Overall
                self.counters['fail_streak_triple'][2] = self.counters['fail_streak_triple'][1]

        #Normalize reward bounds.
        reward = max(min(reward, 2),-2)
        #like c map function:(input val, [inrange_min,inrange_max],[outrange_min,outrange_max]
        reward = np.interp(reward,[-2, 2],
                          [self.params['min_reward'],
                           self.params['max_reward']]) 
        self.counters['feed_reward_que'].append(reward)
        avg_reward = round(np.mean(self.counters['feed_reward_que']),2)

        terminal = False
        """
        lim = self.params['negative_reward_limit']
        if self.counters['reward_polarity'][1] > lim:
            terminal = True
        """

        return avg_reward, repeat_bool, terminal
    
    # A method for communicating with the GUI through Multiprocess Pipe
    def gui_send(self, conn, metrics):
        # just some fodder to signal the end of the transmission.
        metrics.append(-9999)
        for metric in metrics:
            conn.send(metric)

    def logger(self, mode, benchmarks=True):
        #mode: save/load'
        #Pickle is used for counters because deque is not compatible
        #with JSON
        os.chdir(self.home_dir)

        if mode == 'save':

            if benchmarks:
                #Update Benchmarks;
                self.benchmarks_history['training_cycles'] += self.subcycle 
                self.benchmarks_history['training_time'] += self.race_time_delta
                self.benchmarks_history['game_count'] += 1
                if self.completion_bool:
                    self.benchmarks_history['completion_count'] += 1

                self.benchmarks['reward_rate'] = self.counters['reward_rate']
                self.benchmarks['total_reward_trend'] = self.counters['reward_que_avg']
                self.benchmarks['race_time_trend'] = self.counters['race_time_que_avg']
                self.benchmarks['completion_trend'] = round(sum(self.counters['completion_que'])
                                                            /len(self.counters['completion_que'])
                                                            *100,2)
                key = self.benchmarks_history['game_count']
                self.benchmarks_history[key] = self.benchmarks

                ##Save Benchmarks JSON
                benchmarks_json = json.dumps(self.benchmarks_history)
                with open(self.benchmarks_path, 'w') as f:
                    f.write(benchmarks_json)
                    f.close()
            
            ##Save Counters Pickle
            with open(self.counters_path, 'wb') as outp:
                pickle.dump(self.counters, outp, pickle.HIGHEST_PROTOCOL)

        elif mode == 'load':
            ##Load Counters & Benchmarks JSON
            print('Loading Counters Pickle', end="")
            if os.path.exists(self.counters_path):
                with open(self.counters_path, 'rb') as inp:
                    self.counters = pickle.load(inp)
                    print('...done', end="")
            print('Loading Benchmarks JSON',  end="")
            if os.path.exists(self.benchmarks_path):
                with open(self.benchmarks_path) as json_file:
                    loaded_JSON = json.load(json_file)
                    self.benchmarks_history = loaded_JSON
                    #print(loaded_JSON)
                    print('...done')

            self.counters['hist_time'] = self.counters['total_time']

def main(mode, send_connection=False):

    if mode == 'train':

        #start the game
        racers_shortcut_dir = r"Resources\LEGORacers.lnk"
        racers = os.startfile(racers_shortcut_dir)
        time.sleep(9)

        start = datetime.now()
        train_session = ModelTrain(mode)
        train_session.train(start, send_connection) 
        while 1:
            train_session.train(start, send_connection)

    elif mode == 'train_supervised':
        start = datetime.now()
        train_session = ModelTrain(mode)
        train_session.train_supervised(start, send_connection) 
        while 1:
            train_session.train_supervised(start, send_connection)

    elif mode == 'test':
        start = datetime.now()
        test_session = ModelTrain(mode)
        test_session.test(start, send_connection) 
        while 1:
            test_session.test(start, send_connection)

if __name__ == "__main__":
    main('train')
