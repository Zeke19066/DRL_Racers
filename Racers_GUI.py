from cv2 import CAP_DC1394
import numpy as np
import sys
import os
import time
from datetime import datetime, timedelta
import cv2
from collections import deque
from colour import Color
import json

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import pyqtgraph as pg

class MainWindow(qtw.QMainWindow):

    def __init__(self, connection):
        super().__init__()
        self.widget = CentralWidget(connection)
        self.setCentralWidget(self.widget)
        launchcode(self)

class CentralWidget(qtw.QWidget):

    def __init__(self, connection):
        super().__init__()
        self.connection = connection
        self.game_timer = time.time()
        self.master_data_width = 140
        self.fast_mode = False

        self.title_font= qtg.QFont('Trebuchet MS', 30)
        self.metric_font = qtg.QFont('Trebuchet MS', 12)
        self.click_font = qtg.QFont('Trebuchet MS', 20)
        self.metric_font.setItalic(1)
        self.custom_style1 = qtg.QFont('Trebuchet MS', 16)
        self.custom_style2 = qtg.QFont('Trebuchet MS', 12)
        self.custom_style2.setItalic(1)
        self.custom_style3 = qtg.QFont('Trebuchet MS', 16)

        # Load up the graphics and put them in a dictionary.
        self.up_pixmap_1 = qtg.QPixmap('Resources/1_up.png')
        self.up_pixmap_2 = qtg.QPixmap('Resources/2_up.png')
        self.up_pixmap_3 = qtg.QPixmap('Resources/3_up.png')
        self.down_pixmap_1 = qtg.QPixmap('Resources/1_down.png')
        self.down_pixmap_2 = qtg.QPixmap('Resources/2_down.png')
        self.down_pixmap_3 = qtg.QPixmap('Resources/3_down.png')
        self.left_pixmap_1 = qtg.QPixmap('Resources/1_left.png')
        self.left_pixmap_2 = qtg.QPixmap('Resources/2_left.png')
        self.left_pixmap_3 = qtg.QPixmap('Resources/3_left.png')
        self.right_pixmap_1 = qtg.QPixmap('Resources/1_right.png')
        self.right_pixmap_2 = qtg.QPixmap('Resources/2_right.png')
        self.right_pixmap_3 = qtg.QPixmap('Resources/3_right.png')
        self.circle_pixmap_1 = qtg.QPixmap('Resources/1_circle.png')
        self.circle_pixmap_2 = qtg.QPixmap('Resources/2_circle.png')
        self.circle_pixmap_3 = qtg.QPixmap('Resources/3_circle.png')
        self.pixmap_dict = {0:self.up_pixmap_1, 1:self.up_pixmap_2,
                            2:self.up_pixmap_3, 3:self.down_pixmap_1,
                            4:self.down_pixmap_2, 5:self.down_pixmap_3, 
                            6:self.left_pixmap_1, 7:self.left_pixmap_2,
                            8:self.left_pixmap_3, 9:self.right_pixmap_1,
                            10:self.right_pixmap_2,11:self.right_pixmap_3,
                            12:self.circle_pixmap_1, 13:self.circle_pixmap_2,
                            14:self.circle_pixmap_3}
        self.click_set = ['null', 'color: rgb(33, 37, 43);',
                          'color: rgb(255,255,255);',
                          'color: rgb(80,167,239);']
        self.action_set = ['Forward', 'Powerup', 'Reverse', 'Left', 'Right']
        

        self.metrics_last = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 #initialize empty.
                             ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] 

        ## Plot of Racetime
        self.graph_racetime_que1 = deque(maxlen=self.master_data_width)
        self.graph_racetime_que2 = deque(maxlen=self.master_data_width)
        self.graph_racetime_que3 = deque(maxlen=self.master_data_width)
        self.graph_racetime_que4 = deque(maxlen=self.master_data_width)
        self.graph_racetime_lineque = deque(maxlen=self.master_data_width)
        self.graph_racetime = RewardPlotWidget(self, plot_layers=3,
                                               graph_label='Race Time Avg: -',
                                               label_1="Final Time",
                                               label_2="Avg(25)",
                                               label_3="Deviance Abort",
                                               y_min=-5,y_max=300,left=0.045)

        ## Plot of Total Reward
        self.graph_treward_que1 = deque(maxlen=self.master_data_width)
        self.graph_treward_que2 = deque(maxlen=self.master_data_width)
        self.graph_treward_que3 = deque(maxlen=self.master_data_width)
        self.graph_treward_que4 = deque(maxlen=self.master_data_width)
        self.graph_treward_lineque = deque(maxlen=self.master_data_width)
        self.graph_treward = RewardPlotWidget(self, plot_layers=3,
                                              graph_label='Total Reward Avg: -',
                                              label_1="Final Reward",
                                              label_2="Avg(25)",
                                              label_3="Save Abort",
                                              y_min=-5, y_max=1000, left=0.07)

        ## Plot of Avg Actor & Avg Critic Qmax
        self.graph_avg_qmax_que1 = deque(maxlen=self.master_data_width)
        self.graph_avg_qmax_que2 = deque(maxlen=self.master_data_width)
        self.graph_avg_qmax_que3 = deque(maxlen=self.master_data_width)
        self.graph_avg_qmax_que4 = deque(maxlen=self.master_data_width)
        self.graph_avg_qmax = PlotWidget(self,plot_layers=4,
                                         graph_label='Actor vs Critic Avg',
                                         label_1="Agent Avg Qmax",
                                         label_2="Agent Closing Qmax",
                                         label_3="Critic Avg Qmax",
                                         label_4="Critic Closing Qmax",
                                         y_min=-5, y_max=500, left=0.04)

        ## Plot Failrate of moveset
        self.graph_action_failrate_que1 = deque(maxlen=self.master_data_width)
        self.graph_action_failrate_que2 = deque(maxlen=self.master_data_width)
        self.graph_action_failrate_que3 = deque(maxlen=self.master_data_width)
        self.graph_action_failrate_que4 = deque(maxlen=self.master_data_width)
        self.graph_action_failrate = PlotWidget(self,plot_layers=4,
                                                graph_label='Action Failrate',
                                                label_1="Forward",
                                                label_2="Left",
                                                label_3="Right",
                                                label_4="Power-Up",
                                                y_min=-5, y_max=5, left=0.04)

        ## Quickplot of the Actionspace
        num_bars = 5
        self.graph_action_que1 = deque(maxlen=num_bars)
        self.graph_action_que2 = deque(maxlen=num_bars)
        self.graph_action_que3 = deque(maxlen=num_bars)
        self.graph_action_que4 = deque(maxlen=num_bars)
        self.graph_action_que5 = deque(maxlen=num_bars)
        self.graph_action = QuickBarWidget(self,num_bars=num_bars)

        ## Quickplot of the Agent vs Critic vs Reward
        self.graph_actor_critic_que_1 = deque(maxlen=60)
        self.graph_actor_critic_que_2 = deque(maxlen=60)
        self.graph_actor_critic_que_3 = deque(maxlen=60)
        self.graph_actor_critic = QuickCurveWidget(self,minimum=-50, 
                                                   maximum=250)

        ## Quickplot of the Current Dataset Frame.
        self.image_graph = QuickImageWidget(self)

        ## Now populate all of the widgets in the main window.
        self.central_widget_layout = qtw.QGridLayout()
        self.setLayout(self.central_widget_layout)
        self.central_widget_layout.setRowStretch(1, 2) #row, stretch factor. for vertical space per row.
        self.central_widget_layout.setRowStretch(2, 2) #row, stretch factor. for vertical space per row.
        self.central_widget_layout.setRowStretch(3, 2)

        self.central_widget_layout.addWidget(self.title_module(), 0, 0, 1, 18) #(row, column, hight, width)
        self.central_widget_layout.addWidget(self.img_module(self.image_graph), 1, 0, 2, 5) #the image is a frame for the gamewindow

        self.central_widget_layout.addWidget(self.action_module(), 1, 5, 1, 2)
        self.central_widget_layout.addWidget(self.bar_module(self.graph_action), 1, 7, 1, 2)
        self.central_widget_layout.addWidget(self.metrics_module(), 1, 9, 1, 9)
        
        self.central_widget_layout.addWidget(self.curve_module(self.graph_actor_critic), 2, 5, 1, 4)
        self.central_widget_layout.addWidget(self.plot_module(self.graph_treward), 2, 9, 1, 9)

        self.central_widget_layout.addWidget(self.plot_module(self.graph_avg_qmax), 3, 0, 1, 9)
        #self.central_widget_layout.addWidget(self.plot_module(self.graph_racetime),3, 9, 1, 9)
        self.central_widget_layout.addWidget(self.plot_module(self.graph_action_failrate),3, 9, 1, 9)

        self.central_widget_layout.addWidget(self.new_button("quit_button"), 4, 6, 1, 2)
        self.central_widget_layout.addWidget(self.new_button("fast_button"), 4, 8, 1, 2)
        #self.button_click(mode="fast_button") #this will launch in fast mode.


        self.timer = qtc.QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start()

    # This is the mouse arrows & buttons display
    def action_module(self):

        self.up = qtw.QLabel(self)
        self.up.setPixmap(self.pixmap_dict[0])
        self.up.setAlignment(qtc.Qt.AlignCenter)
        #self.up.resize(up_pixmap.width(), up_pixmap.height()) #resize label to window.
        self.down = qtw.QLabel(self)
        self.down.setPixmap(self.pixmap_dict[3])
        self.down.setAlignment(qtc.Qt.AlignCenter)
        self.left = qtw.QLabel(self)
        self.left.setPixmap(self.pixmap_dict[6])
        self.left.setAlignment(qtc.Qt.AlignRight)
        self.right = qtw.QLabel(self)
        self.right.setPixmap(self.pixmap_dict[9])
        self.right.setAlignment(qtc.Qt.AlignLeft)
        self.circle = qtw.QLabel(self)
        self.circle.setPixmap(self.pixmap_dict[12])
        self.circle.setAlignment(qtc.Qt.AlignCenter)

        action_grid = qtw.QGridLayout()
        action_grid.addWidget(self.up, 0, 1,1,1)
        action_grid.addWidget(self.down, 2, 1,1,1)
        action_grid.addWidget(self.left, 1, 0,1,1)
        action_grid.addWidget(self.right, 1, 2,1,1)
        action_grid.addWidget(self.circle, 1, 1,1,1)

        action_groupbox = qtw.QGroupBox()
        action_groupbox.setLayout(action_grid)
        action_groupbox.setFont(self.custom_style2)
        action_groupbox.setStyleSheet(box_style)
        return action_groupbox

    # Tracks training metrics
    def metrics_module(self):
        self.last_move = qtw.QLabel(f' Action: -', self)
        self.last_move.setFont(self.metric_font)
        self.last_move.setStyleSheet(metric_style)
        self.reward_scan = qtw.QLabel(f' Reward Scan: -/-', self)
        self.reward_scan.setFont(self.metric_font)
        self.reward_scan.setStyleSheet(metric_style)
        self.critic_value = qtw.QLabel(f' Actor/Critic Value: -/-', self)
        self.critic_value.setFont(self.metric_font)
        self.critic_value.setStyleSheet(metric_style)
        self.fail_rate = qtw.QLabel(f' Fail Rate: -', self)
        self.fail_rate.setFont(self.metric_font)
        self.fail_rate.setStyleSheet(metric_style)
        self.lap_metrics = qtw.QLabel(f' Lap: -/-', self)
        self.lap_metrics.setFont(self.metric_font)
        self.lap_metrics.setStyleSheet(metric_style)
        self.subcycle_seconds = qtw.QLabel(f' Subcycle/Seconds: -/-', self)
        self.subcycle_seconds.setFont(self.metric_font)
        self.subcycle_seconds.setStyleSheet(metric_style)

        self.current_cycle = qtw.QLabel(f' Current Cycle: -', self)
        self.current_cycle.setFont(self.metric_font)
        self.current_cycle.setStyleSheet(metric_style)
        self.cycles_per_second = qtw.QLabel(f' Cycles/Second: -', self)
        self.cycles_per_second.setFont(self.metric_font)
        self.cycles_per_second.setStyleSheet(metric_style)
        self.avg_qmax = qtw.QLabel(f' Actor/Critic Avg Qmax: -/-', self)
        self.avg_qmax.setFont(self.metric_font)
        self.avg_qmax.setStyleSheet(metric_style)
        self.time_trend = qtw.QLabel(f' Time Trend: -', self)
        self.time_trend.setFont(self.metric_font)
        self.time_trend.setStyleSheet(metric_style)
        self.ai_time = qtw.QLabel(f' Time: -/-', self)
        self.ai_time.setFont(self.metric_font)
        self.ai_time.setStyleSheet(metric_style)
        self.deviance = qtw.QLabel(f' Reward/Second: -/-', self)
        self.deviance.setFont(self.metric_font)
        self.deviance.setStyleSheet(metric_style)
        

        metrics_grid = qtw.QGridLayout()
        metrics_grid.addWidget(self.last_move,1,0,1,1)
        metrics_grid.addWidget(self.reward_scan,2,0,1,2)
        metrics_grid.addWidget(self.critic_value,3,0,1,1)
        metrics_grid.addWidget(self.fail_rate,4,0,1,1)
        metrics_grid.addWidget(self.lap_metrics,5,0,1,1)
        metrics_grid.addWidget(self.subcycle_seconds,6,0,1,1)

        metrics_grid.addWidget(self.current_cycle,1,1,1,1)
        metrics_grid.addWidget(self.cycles_per_second,2,1,1,1)
        metrics_grid.addWidget(self.avg_qmax,3,1,1,1)
        metrics_grid.addWidget(self.time_trend,4,1,1,1)
        metrics_grid.addWidget(self.ai_time,5,1,1,1)
        metrics_grid.addWidget(self.deviance,6,1,1,1)

        metrics_groupbox = qtw.QGroupBox()
        metrics_groupbox.setLayout(metrics_grid)
        metrics_groupbox.setFont(self.custom_style2)
        metrics_groupbox.setStyleSheet(box_style)
        return metrics_groupbox
  
    def update_gui(self):
        self.metrics = []
        #begin_time =time.time()
        try:
            if self.connection.poll(timeout=0.1):
                #print('********************Recieved Connection********************')
                while 1:
                    metric = self.connection.recv()
                    self.metrics.append(metric)
                    if metric == -9999:# dummy figure thrown in to singal the end.
                        break

            #Update Metrics:
            action =  self.metrics[0]
            #this calculates percentages from the other two items in the original entry.
            move_tracker_items = {0:0, 1:0, 2:0, 3:0, 4:0}
            for i in range(5):
                #calculate resulting %
                move_tracker_items[i]= str(round(self.metrics[25][i][0]/self.metrics[25][i][1]*100,2))#+"%"

            #for readability; unpack into variables.
            av = self.metrics[8] #actor value
            av_avg = self.metrics[13] #actor value avg
            cv = self.metrics[11] #critic value
            cv_avg = self.metrics_last[16] #critic value avg
            cc1 = self.metrics[6] #current cycle
            cc2 = self.metrics[20] #gamecount
            rs1 = self.metrics[1]
            rs2 = self.metrics[3]
            rs3 = self.metrics[24][0]
            rs4 = self.metrics[24][1]
            rs5 = round(self.metrics[24][0]-self.metrics[24][1],2)
            ss1 = self.metrics[4]
            ss2 = round(self.metrics[17],1) #Race time sum
            time1 = str(timedelta(seconds=self.metrics[7]))
            time2 = str(timedelta(seconds=self.metrics[15]))
            ttrend1 = self.metrics[10] #avg race time len
            ttrend2 = self.metrics[19] #avg finish time len
            laptxt1 = self.metrics[23][0]
            laptxt2 = round(self.metrics[23][1],1)
            fail_rate = (round(self.metrics[9][1] #[+ count, - count]
                               /sum(self.metrics[9])*100,2)) 
            lr = self.metrics[14][0] #learning Rate
            gb = self.metrics[14][1] #game batch
            epochs = self.metrics[14][2] #epochs
            loss_print = self.metrics[5]
            sub_label = f' Learning Rate: {lr}      Epochs: {epochs}       {loss_print}'
            self.subtitlebar.setText(sub_label)
            #devaince(self.metrics[24]) = [current_total_rewards/current_total_time, avg_total_rewards/avg_total_time, deviance_limit]
            
            self.last_move.setText(f' Action: {self.action_set[action]}')
            self.reward_scan.setText(f' Reward Scan: {rs1}/{rs2}')
            self.deviance.setText(f' Reward/Second: {rs3}/{rs4}/{rs5} delta')
            self.critic_value.setText(f' Actor/Critic Value: {av}/{cv}')
            self.subcycle_seconds.setText(f' Subcycle/Seconds: {ss1}/{ss2}')
            self.fail_rate.setText(f' Fail Rate: {fail_rate}%')
            self.current_cycle.setText(f' Current Cycle: {cc1}/{cc2}')
            self.ai_time.setText(f' Time: {time1}/{time2}')
            self.lap_metrics.setText(f' Lap: {laptxt1}/{laptxt2}')
            self.time_trend.setText(f' Time Trend: {ttrend1}/{ttrend2}')
            self.cycles_per_second.setText(f' Cycles/Second: {self.metrics[12]}')
            self.avg_qmax.setText(f' Actor/Critic Avg Qmax: {av_avg}/{cv_avg}')

            if not self.fast_mode:
                self.reward_scan.setStyleSheet(self.color_coder(1, zero=True)) #Color Code reward_scans

                #update per-frame graphs
                action_arr = self.metrics[2][0] 
                self.graph_action_que1.append(action_arr[0])
                self.graph_action_que2.append(action_arr[1])
                self.graph_action_que3.append(action_arr[2])
                self.graph_action_que4.append(action_arr[3])
                self.graph_action_que5.append(action_arr[4])
                #blocky, but faster than creating a list container
                self.graph_action.add_value(self.graph_action_que1,self.graph_action_que2,
                    self.graph_action_que3,self.graph_action_que4,self.graph_action_que5)

                self.graph_actor_critic_que_1.append(self.metrics[8])
                self.graph_actor_critic_que_2.append(self.metrics[11])
                self.graph_actor_critic_que_3.append(self.metrics[1]*10)

                self.graph_actor_critic.add_value(self.graph_actor_critic_que_1, 
                    self.graph_actor_critic_que_2, self.graph_actor_critic_que_3)
                
                #Actions
                color_set = [1, 1, 1, 1, 1] # All 5 actions start grey, or off state.
                color_set[action] = int(2)
                #print(f'{action}  {color_set[action]}   2')
                self.last_move.setStyleSheet(metric_style)

                #action module color coding. Notice the +/- shifts along pixmap dict selection. 
                up_pixmap = (int(color_set[0]) - 1)
                self.up.setPixmap(self.pixmap_dict[up_pixmap])
                down_pixmap = (int(color_set[2]) + 2)
                self.down.setPixmap(self.pixmap_dict[down_pixmap])
                left_pixmap = (int(color_set[3]) + 5)
                self.left.setPixmap(self.pixmap_dict[left_pixmap])
                right_pixmap = (int(color_set[4]) + 8)
                self.right.setPixmap(self.pixmap_dict[right_pixmap])
                circle_pixmap = (int(color_set[1]) + 11)
                self.circle.setPixmap(self.pixmap_dict[circle_pixmap])

            #Update per-game Graphs
            if ((self.metrics[20] != self.metrics_last[20])
                    and (self.metrics[6] > gb)):
                """## Racetime Update
                self.graph_racetime_que1.append(self.metrics_last[17]) #race time
                self.graph_racetime_que2.append(self.metrics[10]) #avg
                if round(self.metrics_last[24][0]-self.metrics_last[24][1],2) > self.metrics_last[24][2] and self.metrics[20] > 5:#exceeded deviance limit
                    self.graph_racetime_lineque.append(1)
                else:
                    self.graph_racetime_lineque.append(0)  
                graph_racetime_label = f'Race Time Avg: ({self.metrics[10]})'
                self.graph_racetime.add_value(self.graph_racetime_que1, self.graph_racetime_que2, 
                                                self.graph_racetime_lineque, title= graph_racetime_label)
                """

                ## Total Reward Update
                self.graph_treward_que1.append(self.metrics_last[3]) #Final Reward
                self.graph_treward_que2.append(self.metrics[21]) #avg
                if self.metrics_last[3] <= self.metrics[21] and not self.metrics[6]:#Reward was subaverage or we didnt finish
                    self.graph_treward_lineque.append(1)
                else:
                    self.graph_treward_lineque.append(0)
                graph_treward_label = f'Total Reward Avg: ({self.metrics[21]})'
                self.graph_treward.add_value(self.graph_treward_que1, self.graph_treward_que2, 
                                                self.graph_treward_lineque, title = graph_treward_label)

                ## Move Failrate Update
                self.graph_action_failrate_que1.append(move_tracker_items[0]) #Forward
                self.graph_action_failrate_que2.append(move_tracker_items[3]) #Left
                self.graph_action_failrate_que3.append(move_tracker_items[4]) #Right
                self.graph_action_failrate_que4.append(move_tracker_items[1]) #Powerup
                graph_action_failrate_label = f'Action Failrate {move_tracker_items}'
                self.graph_action_failrate.add_value(self.graph_action_failrate_que1, self.graph_action_failrate_que2,
                                                self.graph_action_failrate_que3, self.graph_action_failrate_que4,
                                                title = graph_action_failrate_label)

                ## Qmax/Critic Avg Update
                self.graph_avg_qmax_que1.append(self.metrics_last[13]) #qmax avg
                self.graph_avg_qmax_que2.append(self.metrics_last[8]) #qmax closing
                self.graph_avg_qmax_que3.append(self.metrics_last[16]) #critic avg
                self.graph_avg_qmax_que4.append(self.metrics_last[11]) #critic closing

                graph_avg_qmax_label = f'Actor({self.metrics_last[13]}) vs Critic({self.metrics_last[16]}) Avg'
                self.graph_avg_qmax.add_value(self.graph_avg_qmax_que1, self.graph_avg_qmax_que2,
                                                self.graph_avg_qmax_que3, self.graph_avg_qmax_que4, 
                                                title = graph_avg_qmax_label)


            self.metrics_last = self.metrics
            #print('********************COMPLETED UPDATE GUI********************')
        
        except Exception as e: 
            if str(e) != "'int' object has no attribute 'poll'" and str(e) != "list index out of range":
                print("Receive Failure:",e)
   
    def title_module(self):
        titlebox = qtw.QGroupBox()
        titlebox.setStyleSheet(box_style)
        titlebar = qtw.QLabel(' A2C/PPO Reinforcement Learning Monitor', self)
        titlebar.setFont(self.title_font)
        titlebar.setStyleSheet("color: rgb(255,255,255);border-width : 0px;")
        self.subtitlebar = qtw.QLabel('  v4', self)
        self.subtitlebar.setFont(self.custom_style2)
        self.subtitlebar.setStyleSheet("color: rgb(80,167,239);border-width : 0px;")
        titlegrid = qtw.QGridLayout()
        titlegrid.addWidget(titlebar, 0, 0)
        titlegrid.addWidget(self.subtitlebar, 1, 0)
        titlebox.setLayout(titlegrid)
        return titlebox

    def graph_module(self, graph, label='Graph'):
        graph_grid = qtw.QGridLayout()
        self.q_title = qtw.QLabel(label, self)
        self.q_title.setFont(self.metric_font)
        self.q_title.setStyleSheet(metric_style)
        self.q_title.setAlignment(qtc.Qt.AlignCenter)
        graph_grid.addWidget(self.q_title,0,0)
        graph_grid.addWidget(graph,1,0,5,1)
        graph_groupbox = qtw.QGroupBox()
        graph_groupbox.setLayout(graph_grid)
        graph_groupbox.setFont(self.custom_style2)
        graph_groupbox.setStyleSheet(box_style)
        graph_groupbox.setAlignment(qtc.Qt.AlignCenter)
        return graph_groupbox

    def plot_module(self, plot):
        last_game_groupbox = qtw.QGroupBox()
        last_game_groupbox.setLayout(plot.plot_layout)
        last_game_groupbox.setFont(self.custom_style2)
        last_game_groupbox.setStyleSheet(box_style)
        return last_game_groupbox

    def bar_module(self, plot):
        last_game_groupbox = qtw.QGroupBox()
        last_game_groupbox.setLayout(plot.plot_layout)
        last_game_groupbox.setFont(self.custom_style2)
        last_game_groupbox.setStyleSheet(box_style)
        return last_game_groupbox
    
    def curve_module(self, plot):
        curve_groupbox = qtw.QGroupBox()
        curve_groupbox.setLayout(plot.plot_layout)
        curve_groupbox.setFont(self.custom_style2)
        curve_groupbox.setStyleSheet(box_style)
        return curve_groupbox

    def img_module(self, img_plot):
        img_groupbox = qtw.QGroupBox()
        img_groupbox.setLayout(img_plot.plot_layout)
        img_groupbox.setFont(self.custom_style2)
        img_groupbox.setStyleSheet(box_style)
        return img_groupbox

    def new_button(self, button_request):
        ##Create the button and assign function
        if button_request == "fast_button":
            self.fast_button = qtw.QPushButton("Fast Mode: Off", self)
            self.fast_button.setFont(self.custom_style3)
            self.fast_button.setStyleSheet(click_style_on)
            self.fast_button.clicked.connect(lambda: self.button_click(button_request))
            return self.fast_button

        elif button_request == "quit_button":
            qbtn = qtw.QPushButton('Quit', self)
            qbtn.setFont(self.custom_style3)
            qbtn.setStyleSheet(click_style_on)
            qbtn.clicked.connect(qtw.QApplication.instance().quit)
            return qbtn

    def button_click(self, mode):
        ##Execute Button Function
        if mode == "fast_button":
            #toggle fastmode
            if self.fast_mode:
                self.fast_mode = False
                self.fast_button.setText("Fast Mode: Off")
                self.fast_button.setStyleSheet(click_style_on)
            elif not self.fast_mode:
                self.fast_mode = True
                self.fast_button.setText("Fast Mode: On")
                self.fast_button.setStyleSheet(click_style_off)

    def color_coder(self, metrics_index, distribution=0, percent_max=0, zero=False): # distribution[yellow, orange, red, purple]
        # percent_max will represent 100%, and will trigger percentage mode if provided (0 is off)
        value = self.metrics[metrics_index]
        last_value = self.metrics_last[metrics_index]

        if distribution:
            if percent_max:
                if value/percent_max < distribution[0]:
                    return 'color: rgb(255,255,255);' #Set to normal
                if  distribution[0] <= value/percent_max < distribution[1]:
                    return 'color: rgb(229,192,123);'
                if  distribution[1] <= value/percent_max < distribution[2]:
                    return 'color: rgb(206,101,53);'
                if  distribution[2] <= value/percent_max < distribution[3]:
                    return 'color: rgb(220,89,61);'
                if  value/percent_max >= distribution[3]:
                    return 'color: rgb(198,120,221);'

            elif not percent_max:
                if value < distribution[0]:
                    return 'color: rgb(255,255,255);' #Set to normal
                if  distribution[0] <= value < distribution[1]:
                    return 'color: rgb(229,192,123);'
                if  distribution[1] <= value < distribution[2]:
                    return 'color: rgb(206,101,53);'
                if  distribution[2] <= value < distribution[3]:
                    return 'color: rgb(220,89,61);'
                if  value >= distribution[3]:
                    return 'color: rgb(198,120,221);'

        elif not distribution:
            if zero: #positive/negative
                if value > 0:
                    return 'color: rgb(152,195,121);'
                elif value <= 0:
                    return 'color: rgb(220,89,61);'
            elif not zero: #improve/worsen
                if value >= last_value:
                    return 'color: rgb(152,195,121); border-style: none; text-align: left;'
                elif value < last_value:
                    return 'color: rgb(220,89,61); border-style: none; text-align: left;'

class GraphWidget(qtw.QWidget):
    'A widget to display a runnning graph of information'

    def __init__(self, *args, data_width=100, minimum=0, maximum=100, 
                warn_val=50, crit_val=75, scale=10, **kwargs):
        super().__init__(*args, **kwargs)

        self.maximum, self.minimum  = maximum, minimum
        self.warn_val = warn_val
        self.crit_val = crit_val
        self.scale = scale

        self.bad_color = qtg.QColor(255, 0, 0) #Red
        self.medium_color = qtg.QColor(255, 255, 0) #Yellow
        self.good_color = qtg.QColor(0, 255, 0) #Green

        self.red = Color("red")
        self.lime = Color("lime")
        self.gradient = list(self.red.range_to(self.lime, (self.maximum-self.minimum)))

        self.values = deque([self.minimum]* data_width, maxlen=data_width)
        self.setFixedWidth(data_width * scale)
    
    def add_value(self, value):
        '''
        This method begins by constraining our values between our min and max,
        and then appending it to the deque object
        '''
        value = max(value, self.minimum)
        #value = min(value, self.maximum)
        
        #Dynamic Maximum
        self.maximum-=0.01 #decay the value to keep it dynamic.
        if value > self.maximum:
            self.maximum = value
            self.gradient = list(self.red.range_to(self.lime, (self.maximum-self.minimum)))
        if self.maximum%10==0:
            self.gradient = list(self.red.range_to(self.lime, (self.maximum-self.minimum)))
        
        self.values.append(value)
        self.update()
    
    def paintEvent(self, paint_event):
        painter = qtg.QPainter(self)
        brush = qtg.QBrush(qtg.QColor(48,48,48))
        painter.setBrush(brush)
        painter.drawRect(0, 0, self.width(), self.height())

        pen = qtg.QPen()
        pen.setDashPattern([1,0])

        warn_y = self.val_to_y(self.warn_val)
        pen.setColor(self.good_color)
        painter.setPen(pen)
        painter.drawLine(0, warn_y, self.width(), warn_y)

        crit_y = self.val_to_y(self.crit_val)
        pen.setColor(self.bad_color)
        painter.setPen(pen)
        painter.drawLine(0, crit_y, self.width(), crit_y)

        '''
        gradient = qtg.QLinearGradient(qtc.QPointF(0, self.height()), qtc.QPointF(0, 0))
        gradient.setColorAt(0, self.bad_color)
        gradient.setColorAt(self.warn_val/(self.maximum-self.minimum), self.good_color)
        gradient.setColorAt(self.crit_val/(self.maximum-self.minimum), self.medium_color)

        brush = qtg.QBrush(gradient)
        painter.setBrush(brush)
        painter.setPen(qtc.Qt.NoPen)
        '''

        self.start_value = getattr(self, 'start_value', self.minimum)
        last_value = self.start_value
        self.start_value = self.values[0]

        for i, value in enumerate(self.values):
            local_color = self.gradient[min(int(value),len(self.gradient)-1)]
            local_color = [int(item*255) for item in local_color.rgb]
            #brush = qtg.QBrush(qtg.QColor(str(local_color)))
            brush = qtg.QBrush(qtg.QColor(local_color[0], local_color[1], local_color[2], 150))
            painter.setBrush(brush)
            painter.setPen(qtc.Qt.NoPen)

            x = (i + 1) * self.scale
            last_x = i * self.scale
            y = self.val_to_y(value)
            last_y = self.val_to_y(last_value)

            path = qtg.QPainterPath()
            path.moveTo(x, self.height())
            path.lineTo(last_x, self.height())
            path.lineTo(last_x, last_y)
            #path.lineTo(x, y) #this will draw rectagles, which is more jagged.
            c_x = round(self.scale * 0.5) + last_x
            c1 = (c_x, last_y)
            c2 = (c_x, y)
            path.cubicTo(*c1, *c2, x, y)
            
            painter.drawPath(path)
            last_value = value

    def val_to_y(self, value):
        data_range = self.maximum - self.minimum
        value_fraction = value / data_range
        y_offset = round(value_fraction * self.height())
        y = self.height() - y_offset
        return y

#generic plot module
class PlotWidget(qtw.QWidget):

    def __init__(self, *args,
                plot_layers, graph_label, 
                label_1, label_2='blank', label_3='blank', label_4='blank',
                y_min, y_max, left,
                **kwargs):
        super().__init__(*args, **kwargs)
        
        self.graph_label = graph_label
        self.plot_layers = plot_layers
        self.label_1 = label_1
        self.label_2 = label_2
        self.label_3 = label_3
        self.label_4 = label_4
        self.y_min = y_min
        self.y_max = y_max

        #left=0.6, right=0.985, top=0.935, bottom=0.065 Defaults
        self.canvas = MplCanvas(self, width=1, height=1, dpi=180, left=left, right=0.98,top=0.9, bottom =0.095)
        self.xdata = []
        self.ydata1 = []
        self.ydata2 = []
        self.ydata3 = []
        self.ydata4 = []

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)

        self.add_value([0],[0],[0],[0], self.graph_label)
        
        self.plot_layout = qtw.QVBoxLayout()
        #self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(self.canvas)

    def add_value(self, input_1, input_2=[0], input_3=[0], input_4=[0], title=""):
        self.ydata1 =[float(x) for x in input_1]
        self.ydata2 =[float(x) for x in input_2]
        self.ydata3 =[float(x) for x in input_3]
        self.ydata4 =[float(x) for x in input_4]
        if len(self.ydata1)==1:
            self.ydata1.append(self.ydata1[0])
            self.ydata2.append(self.ydata2[0])
            self.ydata3.append(self.ydata3[0])
            self.ydata4.append(self.ydata4[0])

        self.y_max = round(max( max(self.ydata1), max(self.ydata2), max(self.ydata3), max(self.ydata4) )*1.1) #adjust the limit, with a little space.
        self.y_min = round(min( min(self.ydata1), min(self.ydata2), min(self.ydata3), min(self.ydata4) )*0.9) #adjust the limit, with a little space.
        if int(self.y_max) == int(self.y_min):
            self.y_max+=1
            self.y_min-=1
        #self.xdata.reverse() #we need to do this for label reasons.

        '''
        if len(self.ydata1) < self.n_data: #if our lists arent full (len less than data length)
            self.ydata1 = self.ydata1 + [input_1] 
            self.ydata2 = self.ydata2 + [input_2] 
            self.ydata3 = self.ydata3 + [input_3] 
            self.xdata = list(range(len(self.ydata1))) #x axis will be a list of indexies equal to len(y_axis)
            #self.xdata.reverse() #we need to do this for label reasons.
        elif len(self.ydata1) >= self.n_data: #if we've filled our list que.
            self.ydata1 = self.ydata1[1:] + [input_1] # Drop off the first y element, append a new one.
            self.ydata2 = self.ydata2[1:] + [input_2]
            self.ydata3 = self.ydata3[1:] + [input_3]
            self.xdata = list(range(len(self.ydata1))) #x axis will be a list of indexies equal to len(y_axis)
            #self.xdata.reverse() #we need to do this for label reasons.
        '''
        self.xdata = list(range(len(self.ydata1))) #x axis will be a list of indexies equal to len(y_axis)
        #This is where we set the plot specific style parameters.
        axes = self.canvas.axes
        axes.cla() #clear axis.
        axes.set_facecolor((0.12, 0.12, 0.12))
        axes.set_xlim([0, len(self.ydata1)-1]) #This overwrites the automatic x axis label range.
        #axes.invert_xaxis() #since new values are appended at the end...
        axes.set_ylim([self.y_min, self.y_max]) #This overwrites the automatic y axis label range.
        axes.grid(True)
        axes.tick_params(labelcolor='white')
        axes.set_title(title, color=(1, 1, 1)) #This is how to set title within module
        
        if self.plot_layers == 1:
            axes.plot(self.xdata, self.ydata1, color=(1, 0.4, 1), label=self.label_1)

        elif self.plot_layers == 2:
            legend_PL = mpatches.Patch(color=(1, 0.4, 1), label=self.label_1)
            legend_PL_avg = mpatches.Patch(color=(0, 0.92, 0.83), label=self.label_2)
            axes.legend(handles=[legend_PL, legend_PL_avg])

            axes.plot(self.xdata, self.ydata2, color=(0, 0.92, 0.83))
            axes.plot(self.xdata, self.ydata1, color=(1, 0.4, 1)) #tuple is rgb val

        elif self.plot_layers == 3:
            legend_PL = mpatches.Patch(color=(1, 0.4, 1), label=self.label_1)
            legend_PL_avg = mpatches.Patch(color=(0, 0.92, 0.83), label=self.label_2)
            legend_PL_StDev = mpatches.Patch(color=(0, 1, 0), label=self.label_3)
            axes.legend(handles=[legend_PL, legend_PL_avg, legend_PL_StDev])

            axes.plot(self.xdata, self.ydata3, color=(0, 1, 0), linestyle='--')
            axes.plot(self.xdata, self.ydata2, color=(0, 0.92, 0.83))
            axes.plot(self.xdata, self.ydata1, color=(1, 0.4, 1)) #tuple is rgb val

        elif self.plot_layers == 4:
            legend_PL = mpatches.Patch(color=(1, 0.4, 1), label=self.label_1)
            legend_PL_avg = mpatches.Patch(color=(0, 0.92, 0.83), label=self.label_2)
            legend_PL_StDev = mpatches.Patch(color=(0, 1, 0), label=self.label_3)
            legend_Passive_target = mpatches.Patch(color=(0.90, 0.85, 0.48), label=self.label_4)
            axes.legend(handles=[legend_PL, legend_PL_avg, legend_PL_StDev, legend_Passive_target])

            axes.plot(self.xdata, self.ydata4, color=(0.90, 0.85, 0.48), linestyle='--')
            axes.plot(self.xdata, self.ydata3, color=(0, 1, 0))
            axes.plot(self.xdata, self.ydata2, color=(0, 0.92, 0.83), linestyle='--')
            axes.plot(self.xdata, self.ydata1, color=(1, 0.4, 1)) #tuple is rgb val

        # Trigger the canvas to update and redraw.
        self.canvas.draw()

#bespoke module for reward graph.
class RewardPlotWidget(qtw.QWidget):

    def __init__(self, *args,
                plot_layers, graph_label, 
                label_1, label_2='blank', label_3='blank', label_4='blank',
                y_min, y_max, left,
                **kwargs):
        super().__init__(*args, **kwargs)
        
        self.graph_label = graph_label
        self.plot_layers = plot_layers
        self.label_1 = label_1
        self.label_2 = label_2
        self.label_3 = label_3
        self.label_4 = label_4
        self.y_min = y_min
        self.y_max = y_max

        self.canvas = MplCanvas(self, width=1, height=1, dpi=180, left=left, right=0.98, top=0.92, bottom=0.07)
        self.xdata = []
        self.ydata1 = []
        self.ydata2 = []
        self.ydata3 = []
        self.ydata4 = []

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)
        self.add_value([0],[0],[0],[0], self.graph_label)
        
        self.plot_layout = qtw.QVBoxLayout()
        #self.plot_layout.addWidget(toolbar)

        self.plot_layout.addWidget(self.canvas)

    def add_value(self, input_1, input_2='empty', input_3='empty', input_4='empty', title=""):
        self.ydata1 =[float(x) for x in input_1]
        self.ydata2 =[float(x) for x in input_2]
        #need min of 2 values for function.
        if len(self.ydata1)==1:
            self.ydata1.append(self.ydata1[0])
            self.ydata2.append(self.ydata2[0])

        self.y_max = round(max(self.ydata1)*1.05) #adjust the limit, with a little space.
        self.y_min = round(min(self.ydata1)*0.95) #adjust the limit, with a little space.
        if int(self.y_max) == int(self.y_min):
            self.y_max+=1
            self.y_min-=1

        #this contains 1&0. We want lines where there are 1's.
        self.ydata3 = list(input_3)
        temp = []
        for index, val in enumerate(self.ydata3):
            if val == 1:
                temp.append(index)
        self.ydata3 = temp

        self.ydata4 = list(input_4)
        self.xdata = list(range(len(self.ydata1))) #x axis will be a list of indexies equal to len(y_axis)

        #This is where we set the plot specific style parameters.
        axes = self.canvas.axes
        axes.cla() #clear axis.
        axes.set_facecolor((0.12, 0.12, 0.12))
        axes.set_xlim([0, len(self.ydata1)-1]) #This overwrites the automatic x axis label range.
        #axes.invert_xaxis() #since new values are appended at the end...
        axes.set_ylim([self.y_min, self.y_max]) #This overwrites the automatic y axis label range.
        axes.grid(True)
        axes.tick_params(labelcolor='white')
        axes.set_title(title, color=(1, 1, 1)) #This is how to set title within module
        
        if self.plot_layers == 1:
            axes.plot(self.xdata, self.ydata1, color=(1, 0.4, 1), label=self.label_1)

        elif self.plot_layers == 2:
            legend_PL = mpatches.Patch(color=(1, 0.4, 1), label=self.label_1)
            legend_PL_avg = mpatches.Patch(color=(0, 0.92, 0.83), label=self.label_2)
            axes.legend(handles=[legend_PL, legend_PL_avg])

            axes.plot(self.xdata, self.ydata2, color=(0, 0.92, 0.83))
            axes.plot(self.xdata, self.ydata1, color=(1, 0.4, 1)) #tuple is rgb val

        #y3 is for yellow vertical lines
        elif self.plot_layers == 3:
            legend_PL = mpatches.Patch(color=(1, 0.4, 1), label=self.label_1)
            legend_PL_avg = mpatches.Patch(color=(0, 0.92, 0.83), label=self.label_2)
            legend_PL_StDev = mpatches.Patch(color=(0.90, 0.85, 0.48), label=self.label_3)
            axes.legend(handles=[legend_PL, legend_PL_avg, legend_PL_StDev])

            for vline in self.ydata3:
                axes.axvline(x=vline,  color=(0.90, 0.85, 0.48), linestyle=':')
            #axes.plot(self.xdata, self.ydata3, color=(0.90, 0.85, 0.48), linestyle='--')
            axes.plot(self.xdata, self.ydata2, color=(0, 0.92, 0.83))
            axes.plot(self.xdata, self.ydata1, color=(1, 0.4, 1)) #tuple is rgb val

        elif self.plot_layers == 4:
            legend_PL = mpatches.Patch(color=(1, 0.4, 1), label=self.label_1)
            legend_PL_avg = mpatches.Patch(color=(0, 0.92, 0.83), label=self.label_2)
            legend_PL_StDev = mpatches.Patch(color=(0, 1, 0), label=self.label_3)
            legend_Passive_target = mpatches.Patch(color=(0.8, 0.3, 0.3), label=self.label_4)
            axes.legend(handles=[legend_PL, legend_PL_avg, legend_PL_StDev, legend_Passive_target])

            axes.plot(self.xdata, self.ydata4, color=(0.8, 0.3, 0.3))
            axes.plot(self.xdata, self.ydata3, color=(0, 1, 0), linestyle='--')
            axes.plot(self.xdata, self.ydata2, color=(0, 0.92, 0.83))
            axes.plot(self.xdata, self.ydata1, color=(1, 0.4, 1)) #tuple is rgb val

        # Trigger the canvas to update and redraw.
        self.canvas.draw()

class QuickBarWidget(qtw.QWidget):

    def __init__(self, *args, num_bars=5, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.win = pg.PlotWidget()

        label = 'Action Space'
        title = qtw.QLabel(label, self)
        title.setFont(qtg.QFont('Trebuchet MS', 11))
        title.setStyleSheet(metric_style)
        title.setAlignment(qtc.Qt.AlignCenter)

        sub_label = f"F({5})     P({4})      B({3})     L({2})      R({1})    "
        self.sub_title = qtw.QLabel(sub_label, self)
        self.sub_title.setFont(qtg.QFont('Trebuchet MS', 11))
        self.sub_title.setStyleSheet(metric_style)
        self.sub_title.setAlignment(qtc.Qt.AlignRight)
        self.maximum = 0

        self.red = Color("red")
        self.lime = Color("lime")
        self.gradient = list(self.red.range_to(self.lime, (255)))

        data2 = [[-1],
        [1],
        [-1],
        [1],
        [-1]]

        data3 = [[-1 for n in range(num_bars)],
        [1 for n in range(num_bars)],
        [-1 for n in range(num_bars)],
        [1 for n in range(num_bars)],
        [-1 for n in range(num_bars)]]

        data = [[(np.random.randint(-99,99)/100) for n in range(num_bars)],
        [(np.random.randint(-99,99)/100) for n in range(num_bars)],
        [(np.random.randint(-99,99)/100) for n in range(num_bars)],
        [(np.random.randint(-99,99)/100) for n in range(num_bars)],
        [(np.random.randint(-99,99)/100) for n in range(num_bars)]]


        self.num_bars = num_bars
        #self.bar_line_list = np.array([0,4,8,12,16]) #spaced by 4
        #self.bar_line_list = np.array([0,7,14,21,28]) #spaced by 7
        #self.bar_line_list = np.array([0,1,2,3,4]) #spaced by 7 #Original Tuning, this version yields 5 bar+plots per move
        self.bar_line_list = np.array([n for n in range(num_bars)]) #This will generate dynaic set.

        self.add_value(data[0], data[1], data[2], data[3], data[4])
        
        self.plot_layout = qtw.QVBoxLayout()
        #self.plot_layout = qtw.QGridLayout()
        #self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(title)
        self.plot_layout.addWidget(self.sub_title)
        self.plot_layout.addWidget(self.win)

    def add_value(self, y1, y2, y3, y4, y5):
        yset = [y1, y2, y3, y4, y5]
        ##Catch condition for when the input array is shorter than the bar count per race
        for sub_list in yset:
            while len(sub_list) < self.num_bars:
                #sub_list[:0] = [0] #append to the beggining of the list
                sub_list.insert(0, 0) # second arg is number being inserted

        averages = []
        color_adjust = [1, 0.9, 0.8, 0.7, 0.6]
        self.win.clear()

        averages.append(np.mean(yset[0]))
        averages.append(np.mean(yset[1]))
        averages.append(np.mean(yset[2]))
        averages.append(np.mean(yset[3]))
        averages.append(np.mean(yset[4]))
        places = averages.copy()

        #Dynamic Maximum
        self.maximum-=0.1 #decay the value to keep it dynamic.
        if max(averages) > self.maximum:
            self.maximum = max(averages)

        def color_generator(avg_val, place=None):
            avg_norm = max(avg_val,0) #set zero floor.
            avg_norm = np.interp(avg_norm,[0, self.maximum],[0,254]) #like c map function:(input val, [inrange_min,inrange_max],[outrange_min,outrange_max]
            #avg_norm = avg_norm * color_adjust[place-1]
            color = self.gradient[int(avg_norm)]
            color = [int(item*255) for item in color.rgb]
            ## if value is negative, we darken
            if avg_val<0:
                dark_factor = np.interp(abs(avg_val),[0, self.maximum],[0,254]) #maps from one range to another
                color[0] = max(color[0]-dark_factor,0) #cant be lower than 0
                color[1] = max(color[1]-dark_factor,0)
                color[2] = max(color[2]-dark_factor,0)
 
            return color

        def array_rank(inx):
            # Copy input array into newArray
            input_array = inx
            new_array = input_array.copy()
            # Sort newArray[] in ascending order
            new_array.sort()
            # Dictionary to store the rank of
            # the array element
            ranks = {}
            rank = len(new_array)
            for index in range(len(new_array)):
                element = new_array[index];
                # Update rank of element
                if element not in ranks:
                    ranks[element] = rank
                    rank -= 1
            # Assign ranks to elements
            for index in range(len(input_array)):
                element = input_array[index]
                input_array[index] = ranks[input_array[index]]
            return input_array

        places = array_rank(places)
        y1_color = color_generator(averages[0],places[0])
        y2_color = color_generator(averages[1],places[1])
        y3_color = color_generator(averages[2],places[2])
        y4_color = color_generator(averages[3],places[3])
        y5_color = color_generator(averages[4],places[4])
        self.sub_title.setText(f"F({places[0]})    P({places[1]})     B({places[2]})     L({places[3]})      R({places[4]})    ")

        base = len(self.bar_line_list)

        #Note: the added constant determines the spacing between bar sub-graphs.
        """ Original Calibration
        bg1 = pg.BarGraphItem(x=w,          height=y1, width=1, brush=y1_color)
        bg2 = pg.BarGraphItem(x=w+base+2,   height=y2, width=1, brush=y2_color)
        bg3 = pg.BarGraphItem(x=w+base*2+4, height=y3, width=1, brush=y3_color)
        bg4 = pg.BarGraphItem(x=w+base*3+6, height=y4, width=1, brush=y4_color)
        bg5 = pg.BarGraphItem(x=w+base*4+8, height=y5, width=1, brush=y5_color)
        """

        ##In this version we are trying to increase the plots per move from 5 to 10
        bg1 = pg.BarGraphItem(x=self.bar_line_list,          height=yset[0], width=1, brush=y1_color)
        bg2 = pg.BarGraphItem(x=self.bar_line_list+base+2,   height=yset[1], width=1, brush=y2_color)
        bg3 = pg.BarGraphItem(x=self.bar_line_list+base*2+4, height=yset[2], width=1, brush=y3_color)
        bg4 = pg.BarGraphItem(x=self.bar_line_list+base*3+6, height=yset[3], width=1, brush=y4_color)
        bg5 = pg.BarGraphItem(x=self.bar_line_list+base*4+8, height=yset[4], width=1, brush=y5_color)

        self.win.addItem(bg1)
        self.win.addItem(bg2)
        self.win.addItem(bg3)
        self.win.addItem(bg4)
        self.win.addItem(bg5)

        #self.win.setBackground((30, 30, 30))
        self.win.setBackground((48,48,48))
        self.win.hideAxis('bottom')
        
class QuickCurveWidget(qtw.QWidget):

    def __init__(self, *args, minimum=0, maximum=250, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.win = pg.PlotWidget()

        label = 'Actor vs Critic'
        self.title = qtw.QLabel(label, self)
        self.title.setFont(qtg.QFont('Trebuchet MS', 11))
        self.title.setStyleSheet(metric_style)
        self.title.setAlignment(qtc.Qt.AlignCenter)

        self.maximum, self.minimum = maximum, minimum

        self.actor_pen = pg.mkPen((0, 235, 212), width=3) #blue
        self.critic_pen = pg.mkPen((255, 102, 255), width=3) #violet
        self.reward_pen = pg.mkPen((230, 217, 122), width=3) #yellow
        self.zero_pen = pg.mkPen((175, 175, 175), width=3, style=qtc.Qt.DotLine) #grey dotted

        data = [[10, 20, 30, 40, 50],
        [50, 40, 30, 20, 10],
        [-10, 10, -10, 10, -10]
        ]

        self.add_value(data[0], data[1], data[2])
        
        self.plot_layout = qtw.QVBoxLayout()
        #self.plot_layout = qtw.QGridLayout()
        #self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(self.title)
        self.plot_layout.addWidget(self.win)

    def add_value(self, input_1, input_2, input_3=[]):
        ydata1 = list(input_1)
        ydata2 = list(input_2)
        ydata3 = list(input_3)
        xdata = list(range(len(input_1))) #x axis will be a list of indexies equal to len(y_axis)
        zero_line = [0 for n in xdata]

        label = f'Actor({ydata1[-1]}) vs Critic({ydata2[-1]})'
        self.title.setText(label)

        self.win.clear()

        actor_plot = pg.PlotCurveItem(x=xdata, y=ydata1, pen=self.actor_pen, name="Agent")
        critic_plot =  pg.PlotCurveItem(x=xdata, y=ydata2, pen=self.critic_pen, name="Critic")
        reward_plot =  pg.PlotCurveItem(x=xdata, y=ydata3, pen=self.reward_pen, name="Reward")
        zero_plot =  pg.PlotCurveItem(x=xdata, y=zero_line, pen=self.zero_pen, name="Zero")

        self.win.addItem(actor_plot)
        self.win.addItem(critic_plot)
        self.win.addItem(reward_plot)
        self.win.addItem(zero_plot)

        #self.win.setBackground((30, 30, 30))
        self.win.setBackground((48,48,48))
        self.win.hideAxis('bottom')

class QuickImageWidget(qtw.QWidget):

    def __init__(self, *args, minimum=0, maximum=250, **kwargs):
        super().__init__(*args, **kwargs)

        """
        For speed reasons, we are given the dataset directory through the pipeline, and then load the dataset like the training
        procror does. We tick through "frames" where the index=subcycle in the dataset.
        """
        self.home_dir = os.getcwd()
        square_size= 1000

        ## Create window with GraphicsView widget
        self.win = pg.GraphicsLayoutWidget()
        self.image_object = pg.ImageItem(border='w')
       
        label = '                        Game View'
        self.title = qtw.QLabel(label, self)
        self.title.setFont(qtg.QFont('Trebuchet MS', 16))
        self.title.setStyleSheet(metric_style)
        self.title.setAlignment(qtc.Qt.AlignBottom)

        view = self.win.addViewBox()
        view.setAspectLocked(True) # lock the aspect ratio so pixels are always square
        view.addItem(self.image_object)
        view.setRange(qtc.QRectF(0, 0, square_size, square_size)) ## Set initial view bounds
        view.setGeometry(0, 0, square_size, square_size)
        self.win.setBackground((40,44,52))

        ##Load in the default image.
        #starter_img = cv2.imread("Resources\screenshot_big.png", cv2.IMREAD_UNCHANGED) #cv2.IMREAD_GRAYSCALE
        #self.frame_buffer = []
        #self.frame_buffer.append(starter_img)
        #self.tick_frame(0)

        self.plot_layout = qtw.QVBoxLayout()
        self.plot_layout.addWidget(self.win)
        self.plot_layout.addWidget(self.title)

    #This loads the dataset  
    def change_dataset(self, directory):
        #subfunction for natural sorting of file names.
        import re
        def natural_sorter(data):
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
            return sorted(data, key=alphanum_key)

        #now we're in the folder loading the file list
        os.chdir(self.home_dir)#home first
        os.chdir("Dataset")
        os.chdir(directory)
        self.file_list = os.listdir()
        self.file_list = natural_sorter(self.file_list)

        #preload all the images:
        self.frame_buffer = []
        for file_name in self.file_list:
            sub_frame = cv2.imread(file_name, cv2.IMREAD_UNCHANGED) #cv2.IMREAD_GRAYSCALE
            self.frame_buffer.append(sub_frame)

    #This updates the frame
    def tick_frame(self, subcycle):
        image = self.frame_buffer[subcycle]
        image = np.flip(np.rot90(image))
        self.image_object.setImage(image, autoLevels=None)

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=1, height=4, dpi=100, left=0.6, right=0.985, top=0.935, bottom=0.065):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.set_facecolor((0.16, 0.17, 0.20))
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

def map(x):
    result = ((x-1028)*(100-0)/(430-1028) + 0)
    return result

def launchcode(self):      #the universal 'show' protocol 
    self.setGeometry(0, 0, 3840, 2160)
    #self.setGeometry(1950, 70, 1880, 1600)
    self.setStyleSheet("background-color: rgb(40,44,52);")
    self.setWindowTitle('DRL Monitor')
    #self.show()
    self.showMaximized()

def main(connection):
    #if mode == 'Main_Window':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow(connection,)   
    sys.exit(app.exec())

#margin: 1px;   <- under chunk
pbar_Style1 = """
QProgressBar{
    border: 4px solid grey;
    border-radius: 5px;
    border-color: rgb(152,195,121);
    color: rgb(152,195,121);
    text-align: center
}
QProgressBar::chunk {
    background-color: rgb(152,195,121);
    width: 10px;
}
"""
pbar_Style2 = """
QProgressBar{
    border: 4px solid grey;
    border-radius: 5px;
    border-color: rgb(152,195,121);
    color: rgb(40,44,52);
    text-align: center
}
QProgressBar::chunk {
    background-color: rgb(152,195,121);
    width: 10px;
}
"""

click_style_off ="""
border-color: rgb(255,255,255);
color: rgb(80,167,239);
border-radius: 8px;
border-width : 4px;
border-style:outset;
"""
click_style_on = """
border-color: rgb(255,255,255);
color: rgb(255,255,255);
border-radius: 8px;
border-width : 4px;
border-style:outset;
"""
click_style_random = """
border :3px solid black;
border-color: rgb(209,154,102);
color: rgb(80,167,239);
border-radius: 8px;
border-width: 4px;
"""


#color:rgb(152,195,121); green
#rgb(40,44,52) grey background.

metric_style = """
color: rgb(255,255,255);
border-width : 0px;
border-style: none;
text-align: left;
"""

metric_style_random = """
border-color: rgb(152,195,121);
color: rgb(80,167,239);
border-width : 0px;
border-style: none;
"""

box_style = """
QGroupBox {
    border :3px solid black;
    border-color: rgb(255,255,255);
    color: rgb(209,154,102);
    border-radius: 8px;
    border-width : 4px;
    }
"""

button_style = 0

if __name__ == '__main__':
    main(19)
    # main('Main_Window')