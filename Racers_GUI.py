"""
AoM Training Monitor GUI

Needed:

"""
import numpy as np
import sys
import os
import time
from datetime import datetime, timedelta
from collections import deque
from colour import Color
from collections import deque
import json
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

import pyqtgraph as pg

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc

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
        self.pixmap_dict = {0:self.up_pixmap_1, 1:self.up_pixmap_2, 2:self.up_pixmap_3, 3:self.down_pixmap_1, 4:self.down_pixmap_2, 5:self.down_pixmap_3, 
                            6:self.left_pixmap_1, 7:self.left_pixmap_2, 8:self.left_pixmap_3, 9:self.right_pixmap_1, 10:self.right_pixmap_2,
                            11:self.right_pixmap_3, 12:self.circle_pixmap_1, 13:self.circle_pixmap_2, 14:self.circle_pixmap_3}
        self.click_set = ['null', 'color: rgb(33, 37, 43);', 'color: rgb(255,255,255);', 'color: rgb(80,167,239);']
        self.action_set = ['Forward', 'Powerup', 'Reverse', 'Left', 'Right']
        

        self.metrics_last = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #initialize empty.
        self.central_widget_layout = qtw.QGridLayout()
        self.setLayout(self.central_widget_layout)

        #graph Prepp
        #plt.style.use('dark_background')  #after later style overwrites, this only makes the grid white vs grey.

        self.graph_treward_que1 = deque(maxlen=self.master_data_width)
        self.graph_treward_que1.append(0)
        self.graph_treward_que2 = deque(maxlen=self.master_data_width)
        self.graph_treward_que2.append(0)
        self.graph_treward_que3 = deque(maxlen=self.master_data_width)
        self.graph_treward_que3.append(0)
        self.graph_treward_que4 = deque(maxlen=self.master_data_width)
        self.graph_treward_que4.append(0)

        self.graph_treward_lineque = deque(maxlen=self.master_data_width)
        self.graph_treward_lineque.append(0)

        self.graph_treward = RewardPlotWidget(self, 
            data_width=self.master_data_width, plot_layers=3, graph_label='Final Reward',
            label_1="Final Reward", label_2="Avg(25)", label_3="Learning Session",
            y_min=-5, y_max=1000, left=0.070
            )

        self.graph_avg_qmax_que1 = deque(maxlen=self.master_data_width)
        self.graph_avg_qmax_que1.append(0)
        self.graph_avg_qmax_que2 = deque(maxlen=self.master_data_width)
        self.graph_avg_qmax_que2.append(0)
        self.graph_avg_qmax_que3 = deque(maxlen=self.master_data_width)
        self.graph_avg_qmax_que3.append(0)
        self.graph_avg_qmax_que4 = deque(maxlen=self.master_data_width)
        self.graph_avg_qmax_que4.append(0)
        self.graph_avg_qmax = PlotWidget(self, 
            data_width=self.master_data_width, plot_layers=2, graph_label='Avg Qmax',
            label_1="Avg Qmax", label_2="Closing Qmax",
            y_min=-5, y_max=500, left=0.070
            )
        
        self.graph_action_que1 = deque(maxlen=5)
        self.graph_action_que1.append(1)
        self.graph_action_que2 = deque(maxlen=5)
        self.graph_action_que2.append(1)
        self.graph_action_que3 = deque(maxlen=5)
        self.graph_action_que3.append(1)
        self.graph_action_que4 = deque(maxlen=5)
        self.graph_action_que4.append(1)
        self.graph_action_que5 = deque(maxlen=5)
        self.graph_action_que5.append(1)
        self.graph_action = QuickBarWidget(self,minimum=-150, maximum=250)

        self.graph_actor_critic_que_1 = deque(maxlen=60)
        self.graph_actor_critic_que_1.append(1)
        self.graph_actor_critic_que_2 = deque(maxlen=60)
        self.graph_actor_critic_que_2.append(1)
        self.graph_actor_critic_que_3 = deque(maxlen=60)
        self.graph_actor_critic_que_3.append(1)
        self.graph_actor_critic = QuickCurveWidget(self,minimum=-50, maximum=250)

        self.total_reward_graph = self.graph_treward
        self.action_graph = self.graph_action
        self.actor_critic_graph = self.graph_actor_critic


        """
        self.qmax_graph = GraphWidget(self, data_width=60, minimum=-50, maximum=350, 
        warn_val=100, crit_val=0, scale=10) #no idea why it has to be initialized here, but she do.
        """

        self.central_widget_layout.addWidget(self.title_module(), 0, 0, 1, 6) #(row, column, hight, width)
        self.central_widget_layout.addWidget(self.action_module(), 1, 0, 4, 2)
        self.central_widget_layout.addWidget(self.metrics_module(), 1, 2, 4, 4)
        
        #self.central_widget_layout.addWidget(self.evo_module(), 3, 2, 2, 4)
        self.central_widget_layout.addWidget(self.plot_module(self.graph_avg_qmax), 7, 2, 4, 4)
        #self.central_widget_layout.addWidget(self.graph_module(self.qmax_graph, label='Qmax'), 7, 0, 4, 2)
        self.central_widget_layout.addWidget(self.curve_module(self.actor_critic_graph), 7, 0, 4, 2)
        self.central_widget_layout.addWidget(self.bar_module(self.action_graph),13, 0, 6, 2)
        self.central_widget_layout.addWidget(self.plot_module(self.total_reward_graph),13, 2, 6, 4)

        self.central_widget_layout.addWidget(self.new_button("quit_button"), 19, 2, 1, 1)

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
        self.reward_scan = qtw.QLabel(f' Reward Scan: -', self)
        self.reward_scan.setFont(self.metric_font)
        self.reward_scan.setStyleSheet(metric_style)
        self.critic_value = qtw.QLabel(f' Mistake Tolerance: -', self)
        self.critic_value.setFont(self.metric_font)
        self.critic_value.setStyleSheet(metric_style)
        self.performance_trend = qtw.QLabel(f' Performance Trend: -', self)
        self.performance_trend.setFont(self.metric_font)
        self.performance_trend.setStyleSheet(metric_style)
        self.total_reward = qtw.QLabel(f' Total Reward: -', self)
        self.total_reward.setFont(self.metric_font)
        self.total_reward.setStyleSheet(metric_style)
        self.last_reset = qtw.QLabel(f' Subcycle: -', self)
        self.last_reset.setFont(self.metric_font)
        self.last_reset.setStyleSheet(metric_style)


        self.current_cycle = qtw.QLabel(f' Current Cycle: -', self)
        self.current_cycle.setFont(self.metric_font)
        self.current_cycle.setStyleSheet(metric_style)
        self.cycles_per_second = qtw.QLabel(f' Cycles/Second: -', self)
        self.cycles_per_second.setFont(self.metric_font)
        self.cycles_per_second.setStyleSheet(metric_style)
        self.avg_qmax = qtw.QLabel(f' Avg Qmax: -', self)
        self.avg_qmax.setFont(self.metric_font)
        self.avg_qmax.setStyleSheet(metric_style)
        self.avg_cycles = qtw.QLabel(f' Avg Cycles: -/game', self)
        self.avg_cycles.setFont(self.metric_font)
        self.avg_cycles.setStyleSheet(metric_style)
        self.game_time = qtw.QLabel(f' Game Time: -', self)
        self.game_time.setFont(self.metric_font)
        self.game_time.setStyleSheet(metric_style)
        self.ai_time = qtw.QLabel(f' Time: -', self)
        self.ai_time.setFont(self.metric_font)
        self.ai_time.setStyleSheet(metric_style)
        

        metrics_grid = qtw.QGridLayout()
        metrics_grid.addWidget(self.last_move,1,0,1,1)
        metrics_grid.addWidget(self.reward_scan,2,0,1,2)
        metrics_grid.addWidget(self.performance_trend,3,0,1,1)
        metrics_grid.addWidget(self.critic_value,4,0,1,1)
        metrics_grid.addWidget(self.last_reset,5,0,1,1)
        metrics_grid.addWidget(self.total_reward,6,0,1,1)

        metrics_grid.addWidget(self.current_cycle,1,1,1,1)
        metrics_grid.addWidget(self.cycles_per_second,2,1,1,1)
        metrics_grid.addWidget(self.avg_cycles,3,1,1,1)
        metrics_grid.addWidget(self.avg_qmax,4,1,1,1)
        metrics_grid.addWidget(self.game_time,5,1,1,1)
        metrics_grid.addWidget(self.ai_time,6,1,1,1)

        metrics_groupbox = qtw.QGroupBox()
        metrics_groupbox.setLayout(metrics_grid)
        metrics_groupbox.setFont(self.custom_style2)
        metrics_groupbox.setStyleSheet(box_style)
        return metrics_groupbox

    def evo_module(self):
        self.first_place = qtw.QLabel(f' 1st: -', self)
        self.first_place.setFont(self.metric_font)
        self.first_place.setStyleSheet(metric_style)
        self.second_place = qtw.QLabel(f' 2nd: -', self)
        self.second_place.setFont(self.metric_font)
        self.second_place.setStyleSheet(metric_style)
        self.third_place = qtw.QLabel(f' 3rd: -', self)
        self.third_place.setFont(self.metric_font)
        self.third_place.setStyleSheet(metric_style)        
        self.fourth_place = qtw.QLabel(f' 4th: -', self)
        self.fourth_place.setFont(self.metric_font)
        self.fourth_place.setStyleSheet(metric_style)
        self.fifth_place = qtw.QLabel(f' 5th: -', self)
        self.fifth_place.setFont(self.metric_font)
        self.fifth_place.setStyleSheet(metric_style)

        self.loaded_seed = qtw.QLabel(f' Loaded State: -/-', self)
        self.loaded_seed.setFont(self.metric_font)
        self.loaded_seed.setStyleSheet(metric_style)
        self.current_score = qtw.QLabel(f' Current Score: -/-', self)
        self.current_score.setFont(self.metric_font)
        self.current_score.setStyleSheet(metric_style)
        self.cycles_since_save = qtw.QLabel(f' Last Evo Save: -', self)
        self.cycles_since_save.setFont(self.metric_font)
        self.cycles_since_save.setStyleSheet(metric_style)
        self.standard_deviation = qtw.QLabel(f' StDev: -', self)
        self.standard_deviation.setFont(self.metric_font)
        self.standard_deviation.setStyleSheet(metric_style)
        self.finish_cycles_avg = qtw.QLabel(f' Finish Cycles Avg: -', self)
        self.finish_cycles_avg.setFont(self.metric_font)
        self.finish_cycles_avg.setStyleSheet(metric_style)
        
        evo_grid = qtw.QGridLayout()
        evo_grid.addWidget(self.first_place,1,0,1,1)
        evo_grid.addWidget(self.second_place,2,0,1,1)
        evo_grid.addWidget(self.third_place,3,0,1,1)
        evo_grid.addWidget(self.fourth_place,4,0,1,1)
        evo_grid.addWidget(self.fifth_place,5,0,1,1)

        evo_grid.addWidget(self.loaded_seed,1,1,1,1)
        evo_grid.addWidget(self.current_score,2,1,1,1)
        evo_grid.addWidget(self.cycles_since_save,3,1,1,1)
        evo_grid.addWidget(self.standard_deviation,4,1,1,1)
        evo_grid.addWidget(self.finish_cycles_avg,5,1,1,1)

        evo_groupbox = qtw.QGroupBox()
        evo_groupbox.setLayout(evo_grid)
        evo_groupbox.setFont(self.custom_style2)
        evo_groupbox.setStyleSheet(box_style)
        return evo_groupbox
        
    def update_gui(self):

        self.metrics = []
        color_set = [1, 1, 1, 1, 1] # All 5 actions start grey, or off state.
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

            self.subtitlebar.setText(f' Learning Rate: {self.metrics[14]}      Game Batch: {self.metrics[18]}      Target Avg: {self.metrics[9]}')
    
            self.reward_scan.setText(f' Reward Scan: {self.metrics[1]}')
            self.total_reward.setText(f' Total Reward: {self.metrics[3]}')
            self.critic_value.setText(f' Critic Value: {self.metrics[11]}')
            self.last_reset.setText(f' Subcycle: {self.metrics[4]}')
            self.avg_cycles.setText(f' Avg Subcycles: {self.metrics[5]}')
            self.current_cycle.setText(f' Current Cycle: {self.metrics[6]}/{self.metrics[20]}/{int(self.metrics[20]%self.metrics[18])}')
            self.ai_time.setText(f' Time: {str(timedelta(seconds=self.metrics[7]))}')
            self.game_time.setText(f' Game Time: {str(timedelta(seconds=self.metrics[15]))}')
            self.performance_trend.setText(f' Performance Trend: {self.metrics[10]}')
            self.cycles_per_second.setText(f' Cycles/Second: {self.metrics[12]}')
            self.avg_qmax.setText(f' Avg Qmax: {self.metrics[13]}')

            self.reward_scan.setStyleSheet(self.color_coder(1, zero=True)) #Color Code reward_scans

            #Update Graphs
            if (self.metrics[20] != self.metrics_last[20]) and (self.metrics[6] > self.metrics[18]): #Once per game
                self.graph_treward_que1.append(self.metrics[16]) #performance
                self.graph_treward_que2.append(self.metrics[10]) #avg

                #if (self.metrics[20]-1)%self.metrics[18] == 0:
                if self.metrics[21]:
                    self.graph_treward_lineque.append(1)
                else:
                    self.graph_treward_lineque.append(0)  
                self.total_reward_graph.add_value(self.graph_treward_que1, self.graph_treward_que2, self.graph_treward_lineque)

                self.graph_avg_qmax_que1.append(self.metrics_last[13]) #qmax avg
                self.graph_avg_qmax_que2.append(self.metrics_last[8]) #qmax closing
                self.graph_avg_qmax.add_value(self.graph_avg_qmax_que1, self.graph_avg_qmax_que2)


            action_arr = self.metrics[2][0] 

            #for i, n in enumerate(action_arr):
            #    if n == 0:
            #        action_arr[i] = 1

            self.graph_action_que1.append(action_arr[0])
            self.graph_action_que2.append(action_arr[1])
            self.graph_action_que3.append(action_arr[2])
            self.graph_action_que4.append(action_arr[3])
            self.graph_action_que5.append(action_arr[4])
            #blocky, but faster than creating a list container
            #self.action_graph.add_value(self.graph_action_que1,self.graph_action_que2,
            #    self.graph_action_que3,self.graph_action_que4,self.graph_action_que5)

            
            self.graph_actor_critic_que_1.append(self.metrics[8])
            self.graph_actor_critic_que_2.append(self.metrics[11])
            self.graph_actor_critic_que_3.append(self.metrics[1]*10)

            self.actor_critic_graph.add_value(self.graph_actor_critic_que_1, 
                self.graph_actor_critic_que_2, self.graph_actor_critic_que_3)
            

            '''
            #self.q_title.setText(f' QMax ({self.metrics[8]})')
            #self.qmax_graph.add_value(self.metrics[8])

            self.final_net_worth_trend.setText(f'<font style="rgb(255,255,10);">   Performance Trend: </font><font font style="{str(self.color_coder(9))}">{self.metrics[9]}</font>')
            self.move_reward.setText(f'<font style="rgb(255,255,10);">   Box Radius: </font><font font style="{str(self.color_coder(10))}">{self.metrics[10]}</font>')
            '''

            #Actions
            color_set[action] = int(2)
            #print(f'{action}  {color_set[action]}   2')
            self.last_move.setText(f' Action: {self.action_set[action]}')
            self.last_move.setStyleSheet(metric_style)

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

            self.metrics_last = self.metrics
            #print('********************COMPLETED UPDATE GUI********************')
        
        except Exception as e: 
            if str(e) != "'int' object has no attribute 'poll'" and str(e) != "list index out of range":
                print(e)

        
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

    def new_button(self, button_request):
        if button_request == "update_button":
            update_btn = qtw.QPushButton('Update', self)
            update_btn.setFont(self.custom_style3)
            update_btn.setStyleSheet("""
            border-color: rgb(255,255,255);
            color: rgb(255,255,255);
            border-radius: 8px;
            border-width : 4px;
            border-style:outset;
            """)
            update_btn.clicked.connect(self.buttonClicked)
            return update_btn

        elif button_request == "quit_button":
            qbtn = qtw.QPushButton('Quit', self)
            qbtn.setFont(self.custom_style3)
            qbtn.setStyleSheet("""
            border-color: rgb(255,255,255);
            color: rgb(255,255,255);
            border-radius: 8px;
            border-width : 4px;
            border-style:outset;
            """)
            qbtn.clicked.connect(qtw.QApplication.instance().quit)
            return qbtn

    def buttonClicked(self):
        pass

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
                data_width, plot_layers, graph_label, 
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
        self.n_data = data_width
        self.xdata = []
        self.ydata1 = []
        self.ydata2 = []
        self.ydata3 = []
        self.ydata4 = []

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)

        self.add_value([0],[0],[0],[0])
        
        self.plot_layout = qtw.QVBoxLayout()
        #self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(self.canvas)

    def add_value(self, input_1, input_2=[0], input_3=[0], input_4=[0]):
        self.ydata1 = list(input_1)
        self.ydata2 = list(input_2)
        self.ydata3 = list(input_3)
        self.ydata4 = list(input_4)
        self.y_max = round(max( max(self.ydata1), max(self.ydata2), max(self.ydata3), max(self.ydata4) )*1.1) #adjust the limit, with a little space.
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
        axes.set_xlim([0, self.n_data]) #This overwrites the automatic x axis label range.
        #axes.invert_xaxis() #since new values are appended at the end...
        axes.set_ylim([self.y_min, self.y_max]) #This overwrites the automatic y axis label range.
        axes.grid(True)
        axes.tick_params(labelcolor='white')
        axes.set_title(self.graph_label, color=(1, 1, 1))
        
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
            legend_Passive_target = mpatches.Patch(color=(0.8, 0.3, 0.3), label=self.label_4)
            axes.legend(handles=[legend_PL, legend_PL_avg, legend_PL_StDev, legend_Passive_target])

            axes.plot(self.xdata, self.ydata4, color=(0.8, 0.3, 0.3))
            axes.plot(self.xdata, self.ydata3, color=(0, 1, 0), linestyle='--')
            axes.plot(self.xdata, self.ydata2, color=(0, 0.92, 0.83))
            axes.plot(self.xdata, self.ydata1, color=(1, 0.4, 1)) #tuple is rgb val

        # Trigger the canvas to update and redraw.
        self.canvas.draw()

#bespoke module for reward graph.
class RewardPlotWidget(qtw.QWidget):

    def __init__(self, *args,
                data_width, plot_layers, graph_label, 
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

        self.canvas = MplCanvas(self, width=1, height=1, dpi=180, left=left, right=0.98)
        self.n_data = data_width
        self.xdata = []
        self.ydata1 = []
        self.ydata2 = []
        self.ydata3 = []
        self.ydata4 = []

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)

        self.add_value([0],[0],[0],[0])
        
        self.plot_layout = qtw.QVBoxLayout()
        #self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(self.canvas)

    def add_value(self, input_1, input_2='empty', input_3='empty', input_4='empty'):
        self.ydata1 = list(input_1)
        self.ydata2 = list(input_2)
        self.y_max = round(max(self.ydata1)*1.1) #adjust the limit, with a little space.
        self.y_min = round(min(self.ydata1)) #adjust the limit, with a little space.


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
        axes.set_xlim([0, self.n_data]) #This overwrites the automatic x axis label range.
        #axes.invert_xaxis() #since new values are appended at the end...
        axes.set_ylim([self.y_min, self.y_max]) #This overwrites the automatic y axis label range.
        axes.grid(True)
        axes.tick_params(labelcolor='white')
        axes.set_title(self.graph_label, color=(1, 1, 1))
        
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

    def __init__(self, *args, minimum=0, maximum=250, **kwargs):
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
        self.maximum, self.minimum = maximum, minimum

        self.red = Color("red")
        self.lime = Color("lime")
        self.gradient = list(self.red.range_to(self.lime, (self.maximum-self.minimum)))

        data = [[10, 10, 10, 10, 10],
        [50, 50, 50, 50, 50],
        [100, 100, 100, 100, 100],
        [180, 180, 180, 180, 180],
        [250, 250, 250, 250, 250]]

        data1 = [[10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50]]

        self.add_value(data[0], data[1], data[2], data[3], data[4])
        
        self.plot_layout = qtw.QVBoxLayout()
        #self.plot_layout = qtw.QGridLayout()
        #self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(title)
        self.plot_layout.addWidget(self.sub_title)
        self.plot_layout.addWidget(self.win)

    def add_value(self, y1, y2, y3, y4, y5):

        #w = np.array([0,4,8,12,16]) #spaced by 4
        #w = np.array([0,7,14,21,28]) #spaced by 7
        w = np.array([0,1,2,3,4]) #spaced by 7
        averages = []
        color_adjust = [1, 0.9, 0.8, 0.7, 0.6]
        self.win.clear()

        averages.append(int(np.mean(y1)))
        averages.append(int(np.mean(y2)))
        averages.append(int(np.mean(y3)))
        averages.append(int(np.mean(y4)))
        averages.append(int(np.mean(y5)))


        #Dynamic Maximum
        self.maximum-=0.01 #decay the value to keep it dynamic.
        if max(averages) > self.maximum:
            self.maximum = max(averages)
            self.gradient = list(self.red.range_to(self.lime, (self.maximum-self.minimum)))
        if self.maximum%150==0:
            self.gradient = list(self.red.range_to(self.lime, (self.maximum-self.minimum)))

        
        def color_generator(avg_val, place):
            color = abs(avg_val) * color_adjust[place-1]
            color = self.gradient[int(color)]
            color = [int(item*255) for item in color.rgb]
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

        places = array_rank(averages)
        y1_color = color_generator(averages[0],places[0])
        y2_color = color_generator(averages[1],places[1])
        y3_color = color_generator(averages[2],places[2])
        y4_color = color_generator(averages[3],places[3])
        y5_color = color_generator(averages[4],places[4])

        self.sub_title.setText(f"F({places[0]})    P({places[1]})     B({places[2]})     L({places[3]})      R({places[4]})    ")

        base = len(w)

        #Note: the added constant determines the spacing between bar sub-graphs.
        bg1 = pg.BarGraphItem(x=w,          height=y1, width=1, brush=y1_color)
        bg2 = pg.BarGraphItem(x=w+base+2,   height=y2, width=1, brush=y2_color)
        bg3 = pg.BarGraphItem(x=w+base*2+4, height=y3, width=1, brush=y3_color)
        bg4 = pg.BarGraphItem(x=w+base*3+6, height=y4, width=1, brush=y4_color)
        bg5 = pg.BarGraphItem(x=w+base*4+8, height=y5, width=1, brush=y5_color)

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
    self.setGeometry(1950, 70, 1880, 1980)
    #self.setGeometry(1950, 70, 1880, 1600)
    self.setStyleSheet("background-color: rgb(40,44,52);")
    self.setWindowTitle('DRL Monitor')
    self.show()

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

click_style_off = """
border :3px solid black;
border-color: rgb(33, 37, 43);
color: rgb(33, 37, 43);
border-radius: 8px;
border-width: 4px;
"""
click_style_on = """
border :3px solid black;
border-color: rgb(255,255,255);
color: rgb(255,255,255);
border-radius: 8px;
border-width: 4px;
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