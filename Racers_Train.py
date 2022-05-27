import os
import time
from datetime import datetime
import numpy as np
from collections import deque
from Racers_Agent import Agent

from decorators import function_timer
import Racers_Wrapper

'''
Notes:

Change the Multiprocess pipe terminal signal from -99999 to "END_TRANSMISSION"

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


Batch Mismatch: "Another user came across this bug before. If the number of the peptides is N, then if N % batch_size == 0 or 1, there will be an error:
If N % batch_size == 0, such as in your case N = 15 * x, then the last batch is an empty batch which pytorch cannot handle.
If N % batch_size == 1, i.e. there is only one row in a matrix, then pytorch will treat it differently as a vector."

Learning Rates:
    1e-6 (0.000001) Classic Rate
    1e-7 (0.0000001) Slow Rate

    3e-4 (0.0003) learning rate too fast & unstable
    5e-4 (0.0005)

'''
class ModelTrain():

    def __init__(self, mode):
        print('Initializing Training Model', end="")
        
        self.parameters = { ## User Set Training Hyperperameters & Game Parameters
            ## Hyperparameters:
            'actor_learning_rate':5e-6, #5e-6, 9e-6, 1e-6, 5e-4
            'critic_learning_rate':9e-6, #5e-6, 9e-6, 1e-6, 5e-4
            'gamma': 0.975, #0.95; Discount Factor
            'tau': 0.975, #0.8, 0.65; Scaling parameter for GAE
            'beta':0.005, #0.01; Entropy coefficient; induces more random exploration
            'epsilon': 0.1, #parameter for Clipped Surrogate Objective
            'epochs': 5, #how many times do you want to learn from the data
            'number_of_iterations': 20000000,
            'batch_size': 200, #5; Used for dataset learning.
            'num_local_steps': 800, #20; Used for dataset learning. total learning steps at learn time.

            ## Game Parameters
            'min_reward':-0.5, 'max_reward':0.5, #what the rewards are scaled to
            'reward_target': 750, #sets a minimum reward to save. Can be overwritten by higher avg.
            'reward_deviance_limit': 20, #5; see notes above. Threshold that will triger save-skip.
            'reward_mode':0, #0 is minmap, 1 is speedguage, 2 is show-me.
            'negative_reward_tolerance': 250, #sum TOTAL negative rewards before reset?
            'lap_time_limit': 120, #90; max time per lap before we reset.
            'failure_time_limit': 10, #15; how many seconds of consecutive negative rewards before reset?
            'reward_ratio_limit': 0.75, #0.75; limit of % wrong answers.
            'stuck_limit': 250, #sum negative rewards consecutive before reset?
            'last_place_limit':6000, #limit on consecutive 6th place cycles before reset.
            'off_track_start_1':4, 'off_track_start_2':15, 'off_track_start_3':110, #time in seconds
            'place_bonus':{1:0.2, 2:0.1, 3:0.08, 4:0.06, 5:0.04, 6:0}, #what bonus do we get by position? {1:1.75, 2:1, 3:0.25, 4:0}
            'reset_cycle_limit': 15000, #hard reset every n cycles.
            }

        self.counters = { ## Initialize counters & ques.
            'final_accuracy':0,
            'iteration': 1,
            'game_count':1,
            'reward_que':deque(maxlen=10), #final reward will be a moving avg.
            'side_glitch_que':deque(maxlen=50),
            'current_lap':1,
            'current_lap_time':0,
            'lap_time_dict':{1:0, 2:0, 3:0}, #stores the laptimes
            'game_time':1, #tally of play time in seconds. Menu/reset does not count.
            'stuck_counter': 0, #Counter for consecutive negative rewards.
            'last_place_counter':0,#count consecutive cycles in last place
            'final_reward_score':0, #total reward at last reset.
            'session_move_accuracy': {0:deque([[0,0]],maxlen=25), #each que contains [miss count, total count] for each game
                                    1:deque([[0,0]],maxlen=25), 2:deque([[0,0]],maxlen=25),#note 100 is a stabalizing initial sum
                                    3:deque([[0,0]],maxlen=25), 4:deque([[0,0]],maxlen=25)}, 
            'local_move_accuracy':{0:[0,1], 1:[0,1], 2:[0,1], 3:[0,1], 4:[0,1]}, #[miss count, total count] for current game. Aggregated above^
            
            'reward_performance':deque(maxlen=25),
            'reward_avg_performance': 1,
            'reward_standard_deviation': 0,
            'time_performance':deque(maxlen=25), #Both finish and non-finish races
            'time_avg_performance': 1,
            'time_finish_performance':deque(maxlen=25), #Only counts time of races fully completed (3 laps)
            'time_finish_avg_performance': 0,
            'agent_qmax_list': [],
            'agent_qmax_avg':0,
            'critic_qmax_list': [],
            'critic_qmax_avg':0,
            'reward_deviance': [0, 0], #[reward/time, rewardavg/timeavg]
            
            'reward_polarity': [0,0],#positive count, negative count of rewards.
            'repeat_counter': [0,0], #[current repeat, repeat limit]
            'quit_reset_bool': False,
            'race_over_bool': False,
            'reset_bool':False,
            'parent_counter':1,
            'fail_streak_triple':[0,0,0], #current, session, overall
            'win_count': 0, 'loss_count':0, 'negative_reward_count': 0, #loss is permanent like win, fail is per 'session' and is reset
            }

        self.show_me_stale = 0
        self.cycles_per_second = "Initializing"
        self.file_name = ""
        self.home_dir = os.getcwd()
        self.final_move_accuracy = {0:[0,0], 1:[0,0], 2:[0,0], 
                                    3:[0,0], 4:[0,0]}

        print('... launching Agent', end="")
        self.agent = Agent(self.parameters) #give it the whole dict.
        print('... launching Screen', end="")
        self.screen = Racers_Wrapper.ScreenGrab(dataset_mode="Group") #Group racing
        print('... launching Controller',end="")
        self.controller = Racers_Wrapper.Actions()
        print('... Initialization Complete.')

        if mode != "train_supervised":
            self.controller.first_Run()

    #@function_timer
    def train(self, start, send_connection):

        self.counters['reset_bool'], self.counters['race_over_bool'] = False, False
        self.total_reward, self.subcycle, self.last_reset = 1, 0, self.counters['iteration'] ###
        reward_scan, self.shot_counter, self.race_time_delta, self.total_time = 0, 0, 0, 0  ###
        self.fail_time_start, self.fail_time_bool, self.side_glitch_state_bool = 0, False, False ###
        self.completion_bool = False

        self.agent.initial_state_bool = True
        self.controller.Reset() # Reset the game state for a clean start.
        self.subtime = datetime.now()

        image_data = self.screen.quick_Grab()
        image_data = self.screen.add_watermark(image_data,0) #0 is forward
        image_data = self.screen.resize(image_data, False)
        image_tensor = self.agent.image_to_tensor(image_data)

        ## Main cycle loop
        print('Entering Training Loop')
        while (self.counters['iteration'] < 
                self.parameters['number_of_iterations']):
            
            self.subcycle = max(self.counters['iteration'] 
                                - self.last_reset, 1) 
            action, prob, crit_val, action_space = self.agent.choose_action(image_tensor) #Get NN output ###
            self.controller.current_mapmode = self.parameters['reward_mode']
            self.np_action = action

            ## Execute action
            if ((self.race_time_delta < self.parameters['off_track_start_1']) 
                    and (self.np_action != 0)): #straight at start
                self.controller.action_Map(0) # Override Action
            elif self.np_action == 0:
                 #1/100 chance to fire powerup.
                powerup_rand = np.random.randint(99)
                if not powerup_rand: #on zero
                    self.controller.action_Map(1) # Override Action
                else:
                    self.controller.action_Map(self.np_action) # Agent Action
            else: #Normal Condition
                self.controller.action_Map(self.np_action) # Agent Action
            
            self.image_data_ = self.screen.quick_Grab() # Get the next screenshot
            self.image_data_ = self.screen.add_watermark(self.image_data_,
                                                        self.np_action) #0 is forward

            ## Get meta
            reward_scan, wrong_way_bool, place = self.screen.reward_scan(self.image_data_,
                                                                        self.parameters['reward_mode'],
                                                                        agent_action=self.np_action) # Get Reward
    
            ## Local reward rules
            if self.parameters['reward_mode'] == 0: #Minmap
                sub_reward, terminal = self.local_rules_minmap(reward_scan, 
                                                              wrong_way_bool)
            
            elif self.parameters['reward_mode'] == 1:#Speed
               sub_reward, terminal = self.local_rules_speed(reward_scan, 
                                                            wrong_way_bool,
                                                            place)

            self.image_data_ = self.screen.resize(self.image_data_,
                                                 wrong_way_bool)
            image_tensor_ = self.agent.image_to_tensor(self.image_data_)
            self.agent.remember(image_tensor, action, prob,
                                crit_val, sub_reward, terminal)
            
            image_tensor = image_tensor_
            self.counters['iteration'] += 1

            ## Quit-reset the game once per n cycles to limit glitches
            if self.counters['iteration'] % self.parameters['reset_cycle_limit'] == 0:
                self.counters['quit_reset_bool'] = True

            ## Multiprocessing pipe to send metrics to GUI
            action_space = action_space.detach().cpu().numpy().astype('int32') #ints faster than float.
            action_space=action_space[0]
            qmax = np.max(action_space)
            gui_deviance = (self.counters['reward_deviance']
                            +[self.parameters['reward_deviance_limit']])
            self.counters['agent_qmax_list'].append(qmax)
            self.counters['agent_qmax_avg'] = round(np.mean(self.counters['agent_qmax_list']),1)
            self.counters['critic_qmax_list'].append(crit_val)
            self.counters['critic_qmax_avg'] = round(np.mean(self.counters['critic_qmax_list']),1)
            lap_metrics = [self.counters['current_lap'],self.counters['current_lap_time']]
            action = np.argmax(action_space)
            
            if self.subcycle % 5 == 0: #time calcs are costly, so we limit them
                self.total_time = int((datetime.now() 
                                        - start).total_seconds())
                self.race_time_delta = (datetime.now() 
                                        - self.subtime).total_seconds() #type(time_delta) = Float

            try:
                metrics = [ #Game Metrics
                    action, sub_reward, [action_space], self.total_reward, self.subcycle,                      # 0-4
                    self.counters['time_avg_performance'], self.counters['iteration'], self.total_time, qmax,  # 5-8
                    self.counters['reward_polarity'], self.counters['time_avg_performance'], int(crit_val),    # 9-11
                    self.cycles_per_second, self.counters['agent_qmax_avg'],                                   # 12-13

                    self.parameters['actor_learning_rate'], self.counters['game_time'],                        # 14-15
                    self.counters['critic_qmax_avg'], self.race_time_delta,                                    # 16-17
                    self.parameters['batch_size'], self.counters['time_finish_avg_performance'],               # 18-19
                    self.counters['game_count'], self.counters['reward_avg_performance'],                      # 20,21
                    self.completion_bool, lap_metrics, gui_deviance, self.final_move_accuracy                  # 22-25
                ]

                self.gui_send(send_connection, metrics)
            except Exception as e: 
                print("Send Failure:",e)

            ## Trigger reset; GAME OVER
            if self.counters['reset_bool']:
                self.controller.toggle_pause(True)
                self.training_reset() #learning happens here
                time.sleep(2)
                self.controller.toggle_pause(False)
                return #this will kick us back into the main function below and restart training.

    def train_supervised(self, start, send_connection):
        #store tracking variables.
        self.total_reward, self.subcycle, self.last_reset = 1, 0, self.counters['iteration'] # The action counters
        reward_scan, self.shot_counter = 0, 0

        self.agent.initial_state_bool = True
        self.counters['fail_streak_triple'][1] = 0 #reset the session failstreak

        #Chose the random folder and load the first image. 
        image_data, self.human_action, self.counters['race_over_bool'] = self.screen.data_loader(first_run=True)
        image_data = self.screen.add_watermark_dataset(image_data,0) #0 is forward
        image_tensor = self.agent.image_to_tensor(image_data)
        self.image_data_ = image_data
        self.subtime = datetime.now()

        # main cycle loop
        print('Entering Training Loop')
        while self.counters['iteration'] < self.parameters['number_of_iterations']: 
            self.counters['reset_bool'], self.counters['race_over_bool'] = False, False
            self.subcycle = max(self.counters['iteration'] - self.last_reset, 1)
            self.action, prob, crit_val, action_space = self.agent.choose_action(image_tensor)

            self.np_action = self.action
            reward, repeat_bool = self.local_rules_dataset(self.np_action, self.human_action) # Get Reward
            if not repeat_bool: #only update if we're not repeating
                self.image_data_, self.human_action, self.counters['race_over_bool'] = self.screen.data_loader(first_run=False)
                self.image_data_ = self.screen.add_watermark_dataset(self.image_data_,self.np_action) #0 is forward

            self.counters['reset_bool'] = self.counters['race_over_bool']
            image_tensor_ = self.agent.image_to_tensor(self.image_data_)
            terminal = self.counters['race_over_bool']
            self.agent.remember(image_tensor, self.action, prob, crit_val, reward, terminal)
            
            ## A2C Learning structure
            if self.counters['iteration'] % self.parameters['num_local_steps'] == 0:
                    self.agent.learn()

            image_tensor = image_tensor_
            self.counters['iteration'] += 1

            #Multiprocessing pipe to send metrics to GUI
            action_space = action_space.detach().cpu().numpy().astype('int32') #its faster if we turn it into ints.
            action_space=action_space[0]
            qmax = np.max(action_space)
            value = int(crit_val)
            action = np.argmax(action_space)
            learning_bool = False

            self.total_time = int((datetime.now() - start).total_seconds())
            winrate = round(self.counters['win_count']/self.screen.img_index*100,2) #subcycle would include failed repeats in the accuracy.
            completion_percent = round(self.screen.img_index/len(self.screen.action_list)*100)
            frame_double = [self.screen.img_index,completion_percent]
            image_double = [self.screen.subfolder, self.screen.img_index] #directory, frame#
            self.counters['agent_qmax_list'].append(qmax)
            self.counters['agent_qmax_avg'] = round(np.mean(self.counters['agent_qmax_list']),1)
            self.counters['critic_qmax_list'].append(crit_val)
            self.counters['critic_qmax_avg'] = round(np.mean(self.counters['critic_qmax_list']),1)
            
            #Terminal Printout
            if self.counters['iteration'] % 100 ==0:
                print(f"{self.counters['iteration']} Subcycle:{self.subcycle}/{completion_percent}% W/L:{winrate}%")

            try:
                metrics = [
                    #Game Metrics
                    action, reward, [action_space], winrate, self.subcycle,                                     # 0-4
                    self.counters['time_avg_performance'], self.counters['iteration'], self.total_time, qmax,   # 5-8
                    self.parameters['batch_size'], self.counters['reward_avg_performance'], value,              # 9-11
                    self.cycles_per_second, self.counters['agent_qmax_avg'],                                    # 12-13
                        
                    #Evo Metrics
                    self.parameters['learning_rate'], self.counters['game_time'],                               # 14-15
                    self.counters['final_accuracy'], self.counters['fail_streak_triple'],                       # 16-17
                    self.parameters['batch_size'], self.counters['time_finish_avg_performance'],                # 18-19
                    self.counters['game_count'], self.final_move_accuracy, self.screen.subfolder,               # 20-22
                    frame_double, self.counters['critic_qmax_avg']                                              # 23-24
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
                self.counters['local_move_accuracy'] = {0:[0,1], 1:[0,1], 2:[0,1], 3:[0,1], 4:[0,1]}

                #self.total_reward = max(self.total_reward,10) #set a floor of 10.
                self.counters['game_count'] += 1
                #self.counters['reward_performance'].append(self.total_reward) #let's track how we did.
                self.counters['reward_performance'].append(winrate)
                self.counters['reward_avg_performance'] = round(np.mean(self.counters['reward_performance']),2)
                self.counters['agent_qmax_list'], self.counters['critic_qmax_list'] = [], []
                #Terminal Printout GameOver
                self.counters['final_accuracy'] = round(self.counters['win_count']/self.screen.img_index*100,2)
                print(f"Game Over!         Final Accuracy:{self.counters['final_accuracy']}%         25 Game Avg:{self.counters['reward_avg_performance']}%")

                self.counters['win_count'] = 0
                self.counters['final_reward_score'] = self.total_reward
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

        ## Admission to the winner's circle.
        reward_target = max(self.parameters['reward_target'],
                            self.counters['reward_avg_performance']) #higher of floor or avg
        if self.total_reward > reward_target and self.completion_bool: #only save if we finish
            #save the parent for a later merge.
            merge_name = f"pretrained_model/Merge_Folder/parent{self.counters['parent_counter']}.pth"
            self.agent.save_models(merge_name)
            self.counters['parent_counter'] +=1

        if self.counters['parent_counter'] != 6: #Learn if we're not merging.
            print("Learning......", end="")
            self.agent.learn() #we learn once the race is over.
            print("Done!")
        
        else:#don't learn, the merge is due
            self.agent.memory.clear_memory()
        
        if self.counters['quit_reset_bool']:
            self.counters['quit_reset_bool'] = False
            self.controller.quit_Reset()

        #handle the session_move_accuracy:
        for i in range(5): #For current game
            double = self.counters['local_move_accuracy'][i]
            self.counters['session_move_accuracy'][i].append(double)
        self.counters['local_move_accuracy'] = {0:[0,1], 1:[0,1], 2:[0,1],
                                                3:[0,1], 4:[0,1]}
        self.counters['game_count'] += 1

        self.counters['reward_performance'].append(self.total_reward)
        self.counters['reward_avg_performance'] = round(np.mean(self.counters['reward_performance']),2)
        self.counters['reward_standard_deviation'] = round(np.std(self.counters['reward_performance']),2)
        self.counters['time_performance'].append(self.race_time_delta)
        self.counters['time_avg_performance'] = round(np.mean(self.counters['time_performance']),1)
        if self.completion_bool: #we finished all laps
            self.counters['time_finish_performance'].append(self.race_time_delta) 
            self.counters['time_finish_avg_performance'] = round(np.mean(self.counters['time_finish_performance']),1)

        self.counters['agent_qmax_list'], self.counters['critic_qmax_list'] = [], []
        self.counters['reward_polarity']=[0,0]
        self.counters['current_lap'], self.counters['current_lap_time'] = 1, 0
        self.counters['lap_time_dict'] = {1:0, 2:0, 3:0}

        self.counters['final_reward_score'] = self.total_reward
        self.counters['back_on_track_bonus'] = 0
        self.counters['negative_reward_count'] = 0
        self.last_reset = self.counters['iteration']
        self.subtime = datetime.now()


        if self.counters['parent_counter'] == 6: #time to merge
            self.counters['parent_counter'] = 1 #reset the counter
            self.agent.merge_models()
            #saving here would overwrite the merge^
        else:
            self.agent.save_models()
        return

    ##Rules Block
    def local_rules_minmap(self,reward_scan, wrong_way_bool):
        #Rewerds occur in 2/-2 range and are scaled up do the max/min range.
        terminal, endgame_attempt, side_glitch_bool = False, False, False
        self.final_move_accuracy = {0:[0,0], 1:[0,0], 2:[0,0], 3:[0,0], 4:[0,0]}
        side_set = (2,3,4) #left, right, reverse
        reward, place = reward_scan, 6
        lap_bonus_bool, lap_reward = False, 1

        #combine the aggreagate with the current game.
        for i in range(5): #it's a 5 item number-key dict. For agg
            temp_que = list(self.counters['session_move_accuracy'][i])
            for double in temp_que:
                self.final_move_accuracy[i][0] += double[0]
                self.final_move_accuracy[i][1] += double[1]
        for i in range(5): #For current game
            double = self.counters['local_move_accuracy'][i]
            self.final_move_accuracy[i][0] += double[0]
            self.final_move_accuracy[i][1] += double[1]
        #update the local move counter .......move:[miss count, total count]
        self.counters['local_move_accuracy'][self.np_action][1] += 1 #incriment the move counter for that move

        ##Determine our reward deviance[reward/time, rewardavg/timeavg]
        if self.subcycle>2:
            self.counters['reward_deviance'][0] = round(self.total_reward/self.race_time_delta,2) #reward/timeseconds
            self.counters['reward_deviance'][1] = round(self.counters['reward_avg_performance']/self.counters['time_avg_performance'],2) #reward/timeseconds avg

        ##Determine our lap Time:
        if self.counters['current_lap'] == 1:
            self.counters['current_lap_time'] = self.race_time_delta
        elif self.counters['current_lap'] != 1:
            self.counters['current_lap_time'] = self.race_time_delta-self.counters['lap_time_dict'][self.counters['current_lap']-1]

        ##Determine what lap we're on, once per n cycles.
        if self.counters['current_lap_time'] > 25 and self.subcycle%5 == 0:
            lap_img = self.image_data_[21:72, 632:706]
            lap_check = self.screen.lap_scan(lap_img)
            if lap_check != self.counters['current_lap']: #we are on the next lap.
                self.counters['lap_time_dict'][self.counters['current_lap']] = self.race_time_delta
                self.counters['current_lap'] = lap_check
                lap_bonus_bool = True

        ##See if we've Timed-Out(we get n seconds per lap or reset occurs)
        if self.counters['current_lap_time'] > self.parameters['lap_time_limit']:
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
                side_glitch_percent = round(sum(self.counters['side_glitch_que'])/len(self.counters['side_glitch_que'])*100,1)
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
            powerup_bool = self.screen.item_scan(self.image_data_) #did we have an item available?
            if powerup_bool: #incentivise using powerups.
                reward += 0.5

        if wrong_way_bool:
                reward = -2

        ##No turns at start.
        if self.race_time_delta < self.parameters['off_track_start_1']:  
            if self.np_action in side_set:
                reward = -2
                #endgame_attempt = True

        ##this is for the stuck time & failrate
        if reward <= 0: #Negative Reward
            self.counters['reward_polarity'][1] += 1 #[positive count, negative count]
            self.counters['local_move_accuracy'][self.np_action][0] +=1 #incriment the fail counter for that move.

            #Reset if we've been stuck too long
            if not self.fail_time_bool: #triggering condition
                self.fail_time_bool = True
                self.fail_time_start = self.race_time_delta
            elif self.fail_time_bool and (self.race_time_delta-self.fail_time_start) > self.parameters['failure_time_limit']:
                self.fail_time_bool = False
                endgame_attempt = True
                print(f"Reset: Stuck too long {self.parameters['failure_time_limit']}")

            #if we exceed the failrate
            if (self.counters['reward_polarity'][1]/sum(self.counters['reward_polarity'])) >= self.parameters['reward_ratio_limit']:
                reward = -2 #since agent failed, we want negative association with state.
                endgame_attempt = True

            #Consecutive failures lower the reward incrementally.
            fail_time = self.race_time_delta-self.fail_time_start
            fail_time_reward = (fail_time/15)*1 #maximum reward offset is 1
            reward -= fail_time_reward

        elif reward > 0: #Positive Reward (place-check)
            self.fail_time_bool = False
            self.counters['reward_polarity'][0] += 1 #[positive count, negative count]
            place = self.screen.place_scan(self.image_data_[87:88, 51:87]) #submit the crop

        ##prevent early reset.
        if endgame_attempt and (self.race_time_delta > self.parameters['failure_time_limit'] or self.counters['current_lap']>1): #lap_time_limit failure_time_limit
                terminal = True
                self.counters['reset_bool'] = True

        """##Race time offset. Goal should be about 110seconds
        if self.race_time_delta <= 110 and reward>=0:
            time_reward = self.race_time_delta/110*1 #maximum reward offset is 1, usually we'll be adding a fraction of that.
            reward += time_reward
        elif self.race_time_delta > 110 and reward>=0: #we're taking too long
            #time_reward = self.race_time_delta/110*1 #maximum reward offset is 1, usually we'll be adding a fraction of that.
            reward += 1
        """

        ##Normalize reward bounds.
        reward = max(min(reward, 2),-2)
        reward = np.interp(reward,[-2, 2],[self.parameters['min_reward'],self.parameters['max_reward']]) #like c map function:(input val, [inrange_min,inrange_max],[outrange_min,outrange_max]
        self.counters['reward_que'].append(reward)
        avg_reward = round(np.mean(self.counters['reward_que']),2)

        ##Overrides & adjustments.
        if reward > 0: #positive rewards get a place bonus
            avg_reward += self.parameters['place_bonus'][place] #add in that position bonus; 4th or worse is +0
        if lap_bonus_bool:
            avg_reward = lap_reward
        if side_glitch_bool:
            avg_reward = -0.5

        ##Final Reward Processing.
        self.total_reward += avg_reward
        self.total_reward = round(self.total_reward, 4)
        return avg_reward, terminal
    
    def local_rules_speed(self, reward_scan, wrong_way_bool, place):
        reward = reward_scan
        terminal = False
        endgame_attempt = False
        maximum, minimum = 2, -2

        """# if we got a reward (add in placement bonus e.g. 1st, 2nd etc)
        if reward > 0:
            #reward += self.parameters['place_bonus'][place] #add in that position bonus; 4th or worse is +0
            self.counters['win_count'] += 1

        # if we got a penalty...
        elif reward <= 0:
            pass
        """
        
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
            #endgame_attempt = True

        """#this is for last place contingency
        if place == 6:
            self.counters['last_place_counter'] +=1
            if self.counters['last_place_counter'] >= self.parameters['last_place_limit']:
                self.counters['last_place_counter'] = 0
                endgame_attempt = True
                print("Reset: Too long in last place")
                
        elif place != 6:
            self.counters['last_place_counter'] = 0
        """
        
        reward = round(max(min(reward, maximum), minimum),2) #cap the reward.
        self.total_reward += reward
        self.total_reward = round(self.total_reward,2)

        #this is for the stuck counter. In the Speedometer paradigm, too many negatives mean we're stuck.
        if reward <= 0:
            self.counters['negative_reward_count'] += 1 #abs(reward)
            self.counters['stuck_counter'] += 1 #abs(reward)
            #if we hit the stuck limit
            if self.counters['stuck_counter'] >= self.parameters['stuck_limit']:
                self.counters['stuck_counter'] = 0
                endgame_attempt = True
                #print("Reset: Too many negative rewards")
            #if we hit the total failure limit
            if (self.counters['negative_reward_count'] >= self.parameters['negative_reward_tolerance']): # or (self.avg_yield <= self.counters['reward_cycle_ratio']): # Trigger reset
                reward = minimum #since agent failed, we want negative association with state.
                endgame_attempt = True
            
        elif reward > 0:
            self.counters['stuck_counter']=0

        #prevent early reset.
        if endgame_attempt and self.race_time_delta > 5:
                terminal = True
                self.counters['reset_bool'] = True
        
        return reward, terminal

    def local_rules_dataset(self, agent_action, human_action):
        """
        Notes:
        Right turns are difficult for the agent to learn since they are rare on this track. 
        Incentives should push for addressing this.
        Since they are rare, let's repeat failures on obscure moves so they get more 
            equal representation in the dataset.

        Also, the self.counters['session_move_accuracy'] item has to be tracked as a moving sum
            of the last n games, so that it is dynamic through the entire session, 
            but also doesn't have a hard reset to zeros. 
        This will be achieved through a deque containing lists of the previous game.
        The active game is only contributed to the master count once the game is over and the next
        round starts. The current game's set will be added to the master set while current.
        
        self.action_set = [0'Forward', 1'Powerup', 2'Reverse',
                           3'Left', 4'Right']
        Current Moveset Frequency = {Forward: 55.46%, Powerup: 0.32%, 
                                    Forward: 0.0%, Left: 29.1%,
                                    Right: 15.12%}
    
        """
        first_degree = {0:[1,3,4], 1:[0,3,4],
                        2:[], 3:[0,1], 4:[0,1]} #shows proximity of acceptable alternatives.
        repeat_max = 10 #Maximum number of repeats. actual limit will be fail rate * repeat max. 
        mistake_tolerance = 0.0 #0.20; what % of moves can we miss before a miss triggers a repeat for that move?
        repeat_bool = False

        self.final_move_accuracy = {0:[0,0], 1:[0,0],
                                    2:[0,0], 3:[0,0], 4:[0,0]}
        
        #combine the aggreagate with the current game.
        for i in range(5): #it's a 5 item number-key dict. For agg
            temp_que = list(self.counters['session_move_accuracy'][i])
            for double in temp_que:
                self.final_move_accuracy[i][0] += double[0]
                self.final_move_accuracy[i][1] += double[1]
        for i in range(5): #For current game
            double = self.counters['local_move_accuracy'][i]
            self.final_move_accuracy[i][0] += double[0]
            self.final_move_accuracy[i][1] += double[1]
        #update the local move counter .......move:[miss count, total count]
        self.counters['local_move_accuracy'][human_action][1] += 1 #incriment the move conuter for that move

        #Agent guessed correct (no repeat)
        if agent_action == human_action:
            self.counters['win_count'] +=1
            self.show_me_stale = 0
            self.counters['fail_streak_triple'][0] = 0 #reset the current failstreak
            reward = 2

        #Agent wrong. Let's check for repeats
        elif agent_action != human_action:
            reward = -1

            self.counters['local_move_accuracy'][human_action][0] +=1 #incriment the fail counter for that move.
            #Here we handle repeats
            fail_rate = (self.final_move_accuracy[human_action][0]
                                /self.final_move_accuracy[human_action][1])
            
            #trigger repeat sequence; repeat_counter= [current repeat, repeat limit]
            if fail_rate >= mistake_tolerance:
                reward = -2

                #"""determine how many repeats is limit:
                self.counters['repeat_counter'][1] = round(fail_rate*repeat_max)

                if human_action == 1: #powerup is underrepresented in the dataset
                    self.counters['repeat_counter'][1] = round(fail_rate*repeat_max)*100

                #now handle the repeats
                self.counters['repeat_counter'][0] += 1
                if self.counters['repeat_counter'][0] <= self.counters['repeat_counter'][1]:
                    repeat_bool = True
                elif self.counters['repeat_counter'][0] > self.counters['repeat_counter'][1]:
                    self.counters['repeat_counter'][0] = 0
                    repeat_bool = False
                #"""

            #handle the failstreak triple [current, session, overall]
            self.counters['fail_streak_triple'][0] += 1 #incriment the current failstreak
            if self.counters['fail_streak_triple'][0] > self.counters['fail_streak_triple'][1]: #Current>Sesssion
                self.counters['fail_streak_triple'][1] = self.counters['fail_streak_triple'][0]
                if self.counters['fail_streak_triple'][1] > self.counters['fail_streak_triple'][2]: #Session>Overall
                    self.counters['fail_streak_triple'][2] = self.counters['fail_streak_triple'][1]

        #Normalize reward bounds.
        reward = max(min(reward, 2),-2)
        #like c map function:(input val, [inrange_min,inrange_max],[outrange_min,outrange_max]
        reward = np.interp(reward,[-2, 2],
                          [self.parameters['min_reward'],self.parameters['max_reward']]) 
        self.counters['reward_que'].append(reward)
        avg_reward = round(np.mean(self.counters['reward_que']),2)

        return avg_reward, repeat_bool
       
    # A method for communicating with the GUI through Multiprocess Pipe
    def gui_send(self, conn, metrics):
        # just some fodder to signal the end of the transmission.
        metrics.append(-9999)
        for metric in metrics:
            conn.send(metric)
        
# Script initialization switchboard
def main(mode, send_connection=False):

    if mode == 'train':
        start = datetime.now()
        choo_choo = ModelTrain(mode)
        choo_choo.train(start, send_connection) 
        while 1:
            choo_choo.train(start, send_connection)

    elif mode == 'train_supervised':
        start = datetime.now()
        choo_choo = ModelTrain(mode)
        choo_choo.train_supervised(start, send_connection) 
        while 1:
            choo_choo.train_supervised(start, send_connection)

    elif mode == 'test':
        start = datetime.now()
        proctor = ModelTrain(mode)
        proctor.test(start, send_connection) 
        while 1:
            proctor.test(start, send_connection)

if __name__ == "__main__":
    main('train_supervised')
