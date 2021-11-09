import random
import os
import time
from datetime import datetime
import numpy as np
from collections import deque
from Racers_Agent import Agent

from decorators import function_timer
import Racers_Wrapper_Linear as Racers_Wrapper

'''
Notes:

Another user came across this bug before. If the number of the peptides is N, then if N % batch_size == 0 or 1, there will be an error:
If N % batch_size == 0, such as in your case N = 15 * x, then the last batch is an empty batch which pytorch cannot handle.
If N % batch_size == 1, i.e. there is only one row in a matrix, then pytorch will treat it differently as a vector.

Learning Rates:
    1e-6 (0.000001) Classic Rate
    1e-7 (0.0000001) Slow Rate

    3e-4 (0.0003) learning rate too fast & unstable
    5e-4 (0.0005)

'''
class ModelTrain():

    def __init__(self):
        print('Initializing Training Model')
        
        #dict containing training parameters.
        self.parameters = {
            # Hyperparameters:
            'learning_rate':8e-5, #
            'gamma': 0.95,
            'tau': 0.8,#0.65, #parameter for GAE
            'beta':0.01, #entropy coefficient 0.01
            'epsilon': 0.1, #parameter for Clipped Surrogate Objective
            'epochs': 1, #how many times do you want to learn from the data
            'number_of_iterations': 2000000,
            'game_count_limit': 100, #how many games do we wish to run this for.
            'target_cylce_rate': 6, #Target for cycles per second. we can reduce this by adding a weight that is calculated.
            'batch_size': 5,
            'num_local_steps': 20, #used for first run; overwritten by average subcycles.
            'game_batch': 8, #how many games do we collect for before we learn? Reduces overfitting.

            # Game Parameters
            'performance_target': -250,
            'reward_mode':1,#0 is speedguage, 1 is minmap.
            'negative_reward_tolerance': 250, #sum TOTAL negative rewards before reset?
            'stuck_limit': 250, #sum negative rewards consecutive before reset?
            'last_place_limit':6000, #limit on consecutive 6th place cycles before reset.
            'off_track_start_1':35, 'off_track_start_2':15, 'off_track_start_3':110, 'side_glitch_counter':0, #was 30/45
            'place_bonus':{1:0.2, 2:0.08, 3:0.04, 4:0.02, 6:0}, #what bonus do we get by position? {1:1.75, 2:1, 3:0.25, 4:0}
            'reset_cycle_limit': 15000, #hard reset every n cycles.

            #Placeholders/Counters
            'iteration': 1,
            'game_count':1,
            'game_time':1, #tally of play time in seconds. Menu/reset does not count.
            'stuck_counter': 0, #Counter for consecutive negative rewards.
            'last_place_counter':0,#count consecutive cycles in last place
            'final_reward_score':0, #total reward at last reset.
            'last_save_cycle':0,
            'performance':deque(maxlen=25),
            'cycles_finish_performance':deque(maxlen=25),
            'cycles_performance':deque(maxlen=10), #only across the last 10 races.
            'avg_performance': 0,
            'cycles_finish_avg_performance': 0,
            'cycles_avg_performance': 0,
            'standard_deviation': 0,
            'evo_dict': {},
            'last_save_game': 0, #the game count at the last save checkpoint
            'quit_reset_bool': False,
            'race_over_bool': False,
            'reset_bool':False,
            'win_count': 0, 'loss_count':0, 'negative_reward_count': 0, #loss is permanent like win, fail is per 'session' and is reset
            }

        self.cycles_per_second = "Initializing"
        self.timer_val = 0.067
        self.file_name = ""

        self.agent = Agent(self.parameters) #give it the whole dict.
        self.screen = Racers_Wrapper.ScreenGrab()
        self.controller = Racers_Wrapper.Actions(self.parameters['reward_mode'])
        self.controller.first_Run()

    #@function_timer
    def train(self, start, send_connection):

        #store tracking variables.
        self.total_reward, self.subcycle, self.last_reset = 1, 0, self.parameters['iteration'] # The action counters
        reward_scan, self.shot_counter = 0, 0
        self.subtime = datetime.now()
        q_max_list = []

        self.agent.initial_state_bool = True
        self.controller.Reset() # Reset the game state for a clean start.
        
        #Call to the screenshot & image-to-tensor methods.
        image_data = self.screen.quick_Grab()
        image_data = self.screen.resize(image_data, False)
        image_tensor = self.agent.image_to_tensor(image_data)

        # main cycle loop
        print('Entering Training Loop')
        while self.parameters['iteration'] < self.parameters['number_of_iterations']: 
            self.parameters['reset_bool'], self.parameters['race_over_bool'] = False, False
            self.completion_bool = False
            self.subcycle = max(self.parameters['iteration'] - self.last_reset, 1)
            action, prob, crit_val, action_space = self.agent.choose_action(image_tensor)

            #powerup
            self.np_action = action
            if self.np_action == 1:
                rand = np.random.randint(0,4)
                if rand == 1: #25% chance
                    self.np_action = 0

            self.controller.action_Map(self.np_action) # Execute Action
            
            self.image_data_ = self.screen.quick_Grab() # Get the next screenshot
            #check to see if the race is over.
            if self.subcycle > 1000:
                finish_image = self.image_data_[300-52:440-52,810-104:1220-104]
                self.parameters['race_over_bool'] = self.screen.race_over(finish_image)

            #Get meta
            reward_scan, wrong_way_bool, place = self.screen.reward_scan(self.image_data_, self.parameters['reward_mode']) # Get Reward
    
            # Local reward rules
            if self.parameters['reward_mode'] == 0:
               sub_reward, terminal = self.local_rules_speed(reward_scan, wrong_way_bool, place)

            elif self.parameters['reward_mode'] == 1:
                side_glitch_bool = self.screen.side_glitch(self.image_data_)
                sub_reward, terminal = self.local_rules_minmap(reward_scan, wrong_way_bool, side_glitch_bool)

            self.image_data_ = self.screen.resize(self.image_data_, wrong_way_bool)
            image_tensor_ = self.agent.image_to_tensor(self.image_data_)
            self.agent.remember(image_tensor, action, prob, crit_val, sub_reward, terminal)
            #""" A2C Learning structure
            if self.parameters['iteration'] % self.parameters['num_local_steps'] == 0:
                    #print("LEARNING SESSION.......", end="")
                    #self.controller.toggle_pause(0)
                    self.agent.learn()
                    #self.controller.toggle_pause(1)
                    #time.sleep(0.06)
                    #self.controller.action_Map(self.np_action) # Re Execute Action
                    #print("COMPLETE")
            #"""

            else:
                time.sleep(max(self.timer_val,0))
            
            image_tensor = image_tensor_
            self.parameters['iteration'] += 1

            #Quit-reset the game once per n cycles to limit glitches
            if self.parameters['iteration'] % self.parameters['reset_cycle_limit'] == 0:
                self.parameters['quit_reset_bool'] = True

            #Multiprocessing pipe to send metrics to GUI
            action_space = action_space.detach().cpu().numpy().astype('int32') #its faster if we turn it into ints.
            action_space=action_space[0]
            qmax = np.max(action_space)
            q_max_list.append(qmax)
            q_max_avg = int(np.mean(q_max_list))
            value = int(crit_val)
            action = np.argmax(action_space)
            learning_bool = False
            if self.parameters['avg_performance'] > self.parameters['performance_target']:
                if self.parameters['game_count'] > 25:
                    learning_bool = True
            self.total_time = int((datetime.now() - start).total_seconds())

            try:
                metrics = [
                    #Game Metrics
                    action, sub_reward, [action_space], self.total_reward, self.subcycle,                      # 0-4
                    self.parameters['cycles_avg_performance'], self.parameters['iteration'], self.total_time, qmax,   # 5-8
                    self.parameters['performance_target'], self.parameters['avg_performance'], value,                              # 9-11
                    self.cycles_per_second, q_max_avg,                                                     # 12-13
                        
                    #Evo Metrics
                    self.parameters['learning_rate'], self.parameters['game_time'],                         # 14-15
                    self.parameters['final_reward_score'], 99999999999,                                     # 16-17
                    self.parameters['game_batch'], self.parameters['cycles_finish_avg_performance'],        # 18-19
                    self.parameters['game_count'], learning_bool                                            # 20, 21
                    ]

                self.gui_send(send_connection, metrics)
            except Exception as e: 
                #print out if GUI not operating.
                cycle, tolerance = self.parameters['iteration'], self.parameters['negative_reward_tolerance']
                print(e)
                #print(f'Rwd:{reward_scan} cycle:{cycle} time:{self.adj_time} tol:{tolerance}, total reward:{self.total_reward}, Qmax:{qmax}%, bonus:{move_reward}')

            # Trigger reset; GAME OVER
            if self.parameters['reset_bool']:
                new_time = datetime.now()
                time_delta = (new_time - self.subtime).total_seconds()
                self.parameters['game_time'] += int(time_delta)
                self.cycles_per_second = round(self.subcycle/time_delta,2)
                

                #lets try to adjust to keep a consistant frames per second.
                if self.cycles_per_second > self.parameters['target_cylce_rate']:
                    self.timer_val += 0.001
                elif self.cycles_per_second < self.parameters['target_cylce_rate']:
                    self.timer_val -= 0.001
                    max(self.timer_val, 0.001) #set a floor
                print(f"Timer Val: {self.timer_val}")

                if self.parameters['quit_reset_bool']:
                    self.parameters['quit_reset_bool'] = False
                    self.controller.quit_Reset()

                #self.total_reward = max(self.total_reward,10) #set a floor of 10.
                self.parameters['game_count'] += 1
                self.parameters['performance'].append(self.total_reward) #let's track how we did.
                self.parameters['avg_performance'] = round(np.mean(self.parameters['performance']),2)
                self.parameters['standard_deviation'] = round(np.std(self.parameters['performance']),2)
                self.parameters['cycles_performance'].append(self.subcycle) #let's track how we did.
                self.parameters['cycles_avg_performance'] = round(np.mean(self.parameters['cycles_performance']),1)
                
                if self.completion_bool:
                    self.parameters['cycles_finish_performance'].append(self.subcycle) #let's track how we did.
                    self.parameters['cycles_finish_avg_performance'] = round(np.mean(self.parameters['cycles_finish_performance']),1)

                self.parameters['final_reward_score'] = self.total_reward
                self.total_reward = 0
                self.parameters['side_glitch_counter'] = 0
                self.parameters['back_on_track_bonus'] = 0
                self.parameters['negative_reward_count'] = 0
                self.last_reset = self.parameters['iteration']

                self.subtime = datetime.now()
                
                #self.agent.save_models()

                #only save if we're doing well.
                #if self.parameters['final_reward_score'] >= (self.parameters['performance_target']):
                if self.parameters['final_reward_score'] >= (self.parameters['avg_performance']):
                    self.agent.save_models()
                    if self.parameters['avg_performance'] > self.parameters['performance_target']:
                        if self.parameters['game_count'] > 25:
                            self.parameters['performance_target'] = self.parameters['avg_performance']
                            self.file_name = f"pretrained_model/performance_checkpoint{self.parameters['performance_target']}.pth"
                            self.agent.save_models(file_name=self.file_name)

                """revert to checkpoint if we havent met the target in 100 games.
                if self.parameters['game_count'] - self.parameters['last_save_game'] >= 100:
                    if self.file_name != "":
                        self.agent.load_models(file_name=self.file_name)
                        self.agent.save_models()
                """

                return #this will kick us back into the main function below and restart training.

    # Testing Environment
    def test(self, model):

        '''
        #game_state = GameState()

        # initial action is do nothing
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        action[0] = 1

        image_data, reward, terminal = game_state.frame_step(action)
        image_data = resize_and_rgb2gray(image_data)
        image_data = image_to_tensor(image_data)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

        while True:
            # get output from the neural network
            output = model(state)[0]

            action = torch.zeros([model.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action = action.cuda()

            # get action
            action_index = torch.argmax(output)
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action_index = action_index.cuda()
            action[action_index] = 1

            # get next state
            self.image_data_1, reward, terminal = game_state.frame_step(action)
            self.image_data_1 = resize_and_rgb2gray(self.image_data_1)
            self.image_data_1 = image_to_tensor(self.image_data_1)
            state_1 = torch.cat((state.squeeze(0)[1:, :, :], self.image_data_1)).unsqueeze(0)

            # set state to be state_1
            state = state_1
        '''

    def local_rules_minmap(self,reward_scan, wrong_way_bool, side_glitch_bool):
        terminal = False
        endgame_attempt = False
        side_set = (2,3,4) #left, right, reverse
        reward = reward_scan
        if side_glitch_bool:
            wrong_way_bool = True

        #Check for side glitch
        if self.total_reward > 200:
            #To prevent getting jammed and scoring points on technicality, we'll be checking on the condition if our score is too high.
            if self.subcycle > 650: #check to see if the race is over.
                finish_image = self.image_data_[300-52:440-52,810-104:1220-104]
                self.parameters['race_over_bool'] = self.screen.race_over(finish_image)
                if self.parameters['race_over_bool']:
                    self.completion_bool = True
                    reward = 3
                    endgame_attempt = True

            if self.parameters['iteration'] % 2 == 0:#for efficiency, check only on evens(50%)
                side_glitch_bool = self.screen.side_glitch(self.image_data_)
                if side_glitch_bool:
                    self.parameters['side_glitch_counter'] += 1
                    reward = -1
                    if self.parameters['side_glitch_counter'] > 15:
                        endgame_attempt = True
                        print(f'SIDE GLITCH THRESHOLD! Reward -3')

        # if action = Reverse
        if self.np_action == 2:
            reward = -1

        """
        # No turns at start.
        if self.subcycle < self.parameters['off_track_start_1']:  
            if self.np_action in side_set:
                reward = -2
                endgame_attempt = True
        """

        # if we hit the failure limit.
        #self.avg_yield = round((self.total_reward/self.subcycle)*100)

        if wrong_way_bool:
                reward = -2

        if self.subcycle < (self.parameters['off_track_start_2']): #catch those pesky false negative scores at start.
            if reward < 1:
                reward = 1.2

        self.total_reward += reward
        self.total_reward = round(self.total_reward, 4)

        #this is for the stuck counter. In the Speedometer paradigm, too many negatives mean we're stuck.
        if reward <= 0:
            self.parameters['negative_reward_count'] += abs(reward)
            self.parameters['stuck_counter'] += abs(reward)
            self.parameters['loss_count'] += 1
            #if we hit the stuck limit
            if self.parameters['stuck_counter'] >= self.parameters['stuck_limit']:
                self.parameters['stuck_counter'] = 0
                endgame_attempt = True
                #print("Reset: Too many negative rewards")
            #if we hit the total failure limit
            if (self.parameters['negative_reward_count'] >= self.parameters['negative_reward_tolerance']): # or (self.avg_yield <= self.parameters['reward_cycle_ratio']): # Trigger reset
                reward = -2 #since agent failed, we want negative association with state.
                endgame_attempt = True
            
        elif reward > 0:
            self.parameters['stuck_counter']=0
            self.parameters['win_count'] += 1

        #prevent early reset.
        if endgame_attempt and self.subcycle > 100:
                terminal = True
                self.parameters['reset_bool'] = True

        return reward, terminal
    
    def local_rules_speed(self, reward_scan, wrong_way_bool, place):
        reward = reward_scan
        terminal = False
        endgame_attempt = False
        maximum, minimum = 5, 0

        """# if we got a reward (add in placement bonus e.g. 1st, 2nd etc)
        if reward > 0:
            #reward += self.parameters['place_bonus'][place] #add in that position bonus; 4th or worse is +0
            self.parameters['win_count'] += 1

        # if we got a penalty...
        elif reward <= 0:
            pass
        """
        
        if self.parameters['race_over_bool']: #We finished the race
            self.completion_bool = True
            endgame_attempt = True
            print("Reset: Race is over!")

        if self.subcycle < (self.parameters['off_track_start_2']) and self.np_action != 2: #catch those pesky false negative scores at start.
            if reward <= 0:
                reward = 1
        
        #for reverse actions
        if self.np_action == 2:
            reward = minimum
            #self.parameters['negative_reward_count'] -= 0.5 #just an offset        
        
        #stop spamming shot button (no move)
        if self.np_action == 1:
            self.shot_counter +=1
            #reward -= 5
            if self.shot_counter > 10:
                reward = minimum

        elif self.np_action != 1:
            self.shot_counter = 0

        #incetivize going straight at the start; override pesky cannon shot.
        if self.subcycle < self.parameters['off_track_start_1']:
            reward = max(reward, 0.1) #set a floor
            if self.np_action != 0:
                reward = minimum
            
        #Wrong Way
        if wrong_way_bool:
            #reward -= 0.15
            #reward = min(reward, -0.20)
            reward = minimum
            #endgame_attempt = True

        #this is for last place contingency
        if place == 6:
            self.parameters['last_place_counter'] +=1
            if self.parameters['last_place_counter'] >= self.parameters['last_place_limit']:
                self.parameters['last_place_counter'] = 0
                endgame_attempt = True
                print("Reset: Too long in last place")
                
        elif place != 6:
            self.parameters['last_place_counter'] = 0
        
        #reward = round(max(min(reward, 0.99),-0.99),2) #cap the reward at 1.
        reward = round(max(min(reward, maximum), minimum),2) #cap the reward at 1.
        self.total_reward += reward
        self.total_reward = round(self.total_reward,2)

        #this is for the stuck counter. In the Speedometer paradigm, too many negatives mean we're stuck.
        if reward <= 0:
            self.parameters['negative_reward_count'] += 1 #abs(reward)
            self.parameters['stuck_counter'] += 1 #abs(reward)
            #if we hit the stuck limit
            if self.parameters['stuck_counter'] >= self.parameters['stuck_limit']:
                self.parameters['stuck_counter'] = 0
                endgame_attempt = True
                #print("Reset: Too many negative rewards")
            #if we hit the total failure limit
            if (self.parameters['negative_reward_count'] >= self.parameters['negative_reward_tolerance']): # or (self.avg_yield <= self.parameters['reward_cycle_ratio']): # Trigger reset
                reward = minimum #since agent failed, we want negative association with state.
                endgame_attempt = True
            
        elif reward > 0:
            self.parameters['stuck_counter']=0

        #prevent early reset.
        if endgame_attempt and self.subcycle > 100:
                terminal = True
                self.parameters['reset_bool'] = True
        
        return reward, terminal

    # A method for communicating with the GUI through Multiprocess Pipe
    def gui_send(self, conn, metrics):
        metrics.append(-9999) # just some fodder to signal the end of the transmission.
        for metric in metrics:
            conn.send(metric)
        #conn.close()


# Script initialization switchboard
def main(mode, send_connection=False):

    if mode == 'train':
       
        start = datetime.now()

        choo_choo = ModelTrain()
        choo_choo.train(start, send_connection) #first run
        while 1:

            choo_choo.train(start, send_connection) #first run

if __name__ == "__main__":
    main('train', first_run=False) #Local initialization won't use the 
