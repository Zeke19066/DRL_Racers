"""
AoM Training Monitor GUI

Needed:

"""
import numpy as np
import sys
import os
import time
import multiprocessing
from collections import deque

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc

from psutil import cpu_percent

directory = r'C:\Users\Ezeab\Documents\Python\DRL_AoM\Resources'

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
        self.title_font= qtg.QFont('Trebuchet MS', 30)
        self.metric_font = qtg.QFont('Trebuchet MS', 12)
        self.click_font = qtg.QFont('Trebuchet MS', 20)
        self.metric_font.setItalic(1)
        self.custom_style1 = qtg.QFont('Trebuchet MS', 16)
        self.custom_style2 = qtg.QFont('Trebuchet MS', 12)
        self.custom_style2.setItalic(1)
        self.custom_style3 = qtg.QFont('Trebuchet MS', 16)

        # Load up the graphics and put them in a dictionary.
        self.up_pixmap_1 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/1_up.png')
        self.up_pixmap_2 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/2_up.png')
        self.up_pixmap_3 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/3_up.png')
        self.down_pixmap_1 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/1_down.png')
        self.down_pixmap_2 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/2_down.png')
        self.down_pixmap_3 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/3_down.png')
        self.left_pixmap_1 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/1_left.png')
        self.left_pixmap_2 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/2_left.png')
        self.left_pixmap_3 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/3_left.png')
        self.right_pixmap_1 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/1_right.png')
        self.right_pixmap_2 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/2_right.png')
        self.right_pixmap_3 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/3_right.png')
        self.circle_pixmap_1 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/1_circle.png')
        self.circle_pixmap_2 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/2_circle.png')
        self.circle_pixmap_3 = qtg.QPixmap('C:/Users/Ezeab/Documents/Python/DRL_AoM/Resources/3_circle.png')
        self.pixmap_dict = {0:self.up_pixmap_1, 1:self.up_pixmap_2, 2:self.up_pixmap_3, 3:self.down_pixmap_1, 4:self.down_pixmap_2, 5:self.down_pixmap_3, 
                            6:self.left_pixmap_1, 7:self.left_pixmap_2, 8:self.left_pixmap_3, 9:self.right_pixmap_1, 10:self.right_pixmap_2,
                            11:self.right_pixmap_3, 12:self.circle_pixmap_1, 13:self.circle_pixmap_2, 14:self.circle_pixmap_3}
        self.click_set = ['null', 'color: rgb(33, 37, 43);', 'color: rgb(255,255,255);', 'color: rgb(80,167,239);']
        self.action_set = ['Do Nothing', 'Right Click', 'Left Click', 'Move Left', 'Move Right', 'Move Up', 'Move Down']
        
        
    
        self.metrics_last = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #initialize empty.
        layout = qtw.QGridLayout()
        self.setLayout(layout)
        #self.graph_instance = GraphWidget(self) #no idea why it has to be initialized here, but she do.
        layout.addWidget(self.title_module(), 0, 0, 1, 3) #(row, column, hight, width)
        layout.addWidget(self.action_tracker(), 1, 0, 1, 1)
        layout.addWidget(self.metrics_module(), 1, 1, 1, 2)
        #layout.addWidget(self.graph_module(), 2, 0)
        layout.addWidget(self.new_button("quit_button"), 4, 1, 2, 1)
        
        self.timer = qtc.QTimer()
        self.timer.setInterval(1)
        #self.timer.timeout.connect(self.update_graph)
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start()

    # This is the mouse arrows & buttons display
    def action_tracker(self):

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
        self.left_click = qtw.QLabel(f'Left Click', self)
        self.left_click.setFont(self.click_font)
        self.left_click.setStyleSheet(click_style_off)
        self.right_click = qtw.QLabel(f'Right Click', self)
        self.right_click.setFont(self.click_font)
        self.right_click.setStyleSheet(click_style_off)

        action_grid = qtw.QGridLayout()
        action_grid.addWidget(self.up, 0, 1)
        action_grid.addWidget(self.down, 2, 1)
        action_grid.addWidget(self.left, 1, 0)
        action_grid.addWidget(self.right, 1, 2)
        action_grid.addWidget(self.circle, 1, 1)
        action_grid.addWidget(self.left_click, 3, 0)
        action_grid.addWidget(self.right_click, 3, 2)
        action_groupbox = qtw.QGroupBox()
        action_groupbox.setLayout(action_grid)
        action_groupbox.setFont(self.custom_style2)
        action_groupbox.setStyleSheet(box_style)
        return action_groupbox

    # Tracks training metrics
    def metrics_module(self):
        self.last_move = qtw.QLabel(f' Last Move: -', self)
        self.last_move.setFont(self.metric_font)
        self.last_move.setStyleSheet(metric_style)
        self.combo = qtw.QLabel(f' Combo Count: -', self)
        self.combo.setFont(self.metric_font)
        self.combo.setStyleSheet(metric_style)
        self.performance_trend = qtw.QLabel(f' Performance Trend: - ', self)
        self.performance_trend.setFont(self.metric_font)
        self.performance_trend.setStyleSheet(metric_style)
        self.box_size = qtw.QLabel(f' Box Radius: - ', self)
        self.box_size.setFont(self.metric_font)
        self.box_size.setStyleSheet(metric_style)
        self.reward_efficiency = qtw.QLabel(f' Reward Probability: - / -', self)
        self.reward_efficiency.setFont(self.metric_font)
        self.reward_efficiency.setStyleSheet(metric_style)
        self.total_reward = qtw.QLabel(f' Total Reward: -', self)
        self.total_reward.setFont(self.metric_font)
        self.total_reward.setStyleSheet(metric_style)

        self.last_reset = qtw.QLabel(f' Cycles Past Reset: -', self)
        self.last_reset.setFont(self.metric_font)
        self.last_reset.setStyleSheet(metric_style)
        self.win_count = qtw.QLabel(f' Win Count: -', self)
        self.win_count.setFont(self.metric_font)
        self.win_count.setStyleSheet(metric_style)
        self.fail_count = qtw.QLabel(f' Fail Count: -', self)
        self.fail_count.setFont(self.metric_font)
        self.fail_count.setStyleSheet(metric_style)
        self.epsilon = qtw.QLabel(f' Epsilon: -', self)
        self.epsilon.setFont(self.metric_font)
        self.epsilon.setStyleSheet(metric_style)
        self.current_cycle = qtw.QLabel(f' Current Cycle: -', self)
        self.current_cycle.setFont(self.metric_font)
        self.current_cycle.setStyleSheet(metric_style)
        self.qmax = qtw.QLabel(f' QMax: -', self)
        self.qmax.setFont(self.metric_font)
        self.qmax.setStyleSheet(metric_style)
        self.ai_time = qtw.QLabel(f' Time: - / -', self)
        self.ai_time.setFont(self.metric_font)
        self.ai_time.setStyleSheet(metric_style)
        

        metrics_grid = qtw.QGridLayout()
        metrics_grid.addWidget(self.last_move,1,0,1,1)
        metrics_grid.addWidget(self.combo,2,0,1,2)
        metrics_grid.addWidget(self.performance_trend,3,0,1,1)
        metrics_grid.addWidget(self.box_size,4,0,1,1)
        metrics_grid.addWidget(self.last_reset,5,0,1,1)
        metrics_grid.addWidget(self.reward_efficiency,6,0,1,1)
        metrics_grid.addWidget(self.total_reward,7,0,1,1)

        metrics_grid.addWidget(self.current_cycle,1,1,1,1)
        metrics_grid.addWidget(self.win_count,2,1,1,1)
        metrics_grid.addWidget(self.fail_count,3,1,1,1)
        metrics_grid.addWidget(self.epsilon,4,1,1,1)
        metrics_grid.addWidget(self.qmax,5,1,1,1)
        metrics_grid.addWidget(self.ai_time,6,1,1,1)

        metrics_groupbox = qtw.QGroupBox()
        metrics_groupbox.setLayout(metrics_grid)
        metrics_groupbox.setFont(self.custom_style2)
        metrics_groupbox.setStyleSheet(box_style)
        return metrics_groupbox

    def update_metrics(self):
        self.metrics = []
        color_set = [1, 1, 1, 1, 1, 1, 1] # All 7 actions start grey, or off state.
        #begin_time =time.time()
        
        try:
            while 1:
                metric = self.connection.recv()
                self.metrics.append(metric)
                if metric == -9999:# dummy figure thrown in to singal the end.
                    break
            
            #Metrics int(self.metrics[0][0])
            action, rnd_qmax, reward_prob = int(self.metrics[0]), round(float(self.metrics[7]), 2), round(((int(self.metrics[2])/int(self.metrics[5]))*100), 1)
            adj_reward_prob = round(( (int(self.metrics[2]) / (int(self.metrics[5])/(self.metrics[10]/5)) ) * 100), 1) #assuming only box_radius/5 maximum efficiency.
            adj_time = round(float(self.metrics[6]/60),2)
            self.combo.setText(f' Combo Count: {self.metrics[1]}/{round((self.metrics[10]/5))}')
            self.total_reward.setText(f' Total Reward: {self.metrics[2]}')
            self.last_reset.setText(f' Last Reset: {self.metrics[3]}')
            self.epsilon.setText(f' Epsilon: {self.metrics[4]*100}%')
            self.current_cycle.setText(f' Current Cycle: {self.metrics[5]}')
            self.ai_time.setText(f' Time: {self.metrics[6]}sec/{adj_time}min')
            self.qmax.setText(f' QMax: {rnd_qmax*100}%')
            self.reward_efficiency.setText(f' Reward Efficiency: {reward_prob}%/{adj_reward_prob}%')
            
            self.performance_trend.setText(f' Performance Trend: {self.metrics[9]}')
            self.box_size.setText(f' Box Radius: {self.metrics[10]}px')
            
            self.win_count.setText(f' Win Count: {self.metrics[11]}/{round((self.metrics[11]/(self.metrics[11]+self.metrics[12])*100), 2)}%')
            self.fail_count.setText(f' Fail Count: {self.metrics[12]}/{round((self.metrics[12]/(self.metrics[11]+self.metrics[12])*100), 2)}%')


            self.combo.setStyleSheet(self.color_coder(1, [0.5, 0.6, 0.75, 1], percent_max=(self.metrics[10]/5))) #Color Code Combos
            self.last_reset.setStyleSheet(self.color_coder(3, [0.20, 0.35, 0.50, 0.75], percent_max=80)) #Color Code Last Reset
            
            '''
            self.performance_trend.setText(f'<font style="rgb(255,255,10);">   Performance Trend: </font><font font style="{str(self.color_coder(9))}">{self.metrics[9]}</font>')
            self.box_size.setText(f'<font style="rgb(255,255,10);">   Box Radius: </font><font font style="{str(self.color_coder(10))}">{self.metrics[10]}</font>')
            '''

            if self.metrics[8]:
                color_set[action] = int(3)
                #print(f'{action}  {color_set[action]}   3')
                self.last_move.setText(f' Random Move: {self.action_set[action]}')
                self.last_move.setStyleSheet(metric_style_random)
            
            elif not self.metrics[8]:
                color_set[action] = int(2)
                #print(f'{action}  {color_set[action]}   2')
                self.last_move.setText(f' Last Move: {self.action_set[action]}')
                self.last_move.setStyleSheet(metric_style)

            #Actions
            up_pixmap = (int(color_set[5]) - 1)
            self.up.setPixmap(self.pixmap_dict[up_pixmap])
            down_pixmap = (int(color_set[6]) + 2)
            self.down.setPixmap(self.pixmap_dict[down_pixmap])
            left_pixmap = (int(color_set[3]) + 5)
            self.left.setPixmap(self.pixmap_dict[left_pixmap])
            right_pixmap = (int(color_set[4]) + 8)
            self.right.setPixmap(self.pixmap_dict[right_pixmap])
            circle_pixmap = (int(color_set[0]) + 11)
            self.circle.setPixmap(self.pixmap_dict[circle_pixmap])
            right_val = color_set[1]
            self.right_click.setStyleSheet(self.click_set[right_val])
            left_val = color_set[2]
            self.left_click.setStyleSheet(self.click_set[left_val])

            self.metrics_last = self.metrics

        except Exception as e: print(e)
    # Graph Section
    def update_graph(self):
        cpu_usage = cpu_percent()
        self.graph_instance.add_value(cpu_usage)

    def title_module(self):
        titlebox = qtw.QGroupBox()
        titlebox.setStyleSheet(box_style)
        titlebar = qtw.QLabel('Deep Reinforcement Learning Monitor', self)
        titlebar.setFont(self.title_font)
        titlebar.setStyleSheet("color: rgb(255,255,255);border-width : 0px;")
        subtitlebar = qtw.QLabel('v1.0 Beta', self)
        subtitlebar.setFont(self.custom_style2)
        subtitlebar.setStyleSheet("color: rgb(80,167,239);border-width : 0px;")
        titlegrid = qtw.QGridLayout()
        titlegrid.addWidget(titlebar, 0, 0)
        titlegrid.addWidget(subtitlebar, 1, 0)
        titlebox.setLayout(titlegrid)
        return titlebox

    def graph_module(self):
        graph_grid = qtw.QGridLayout()
        graph_grid.addWidget(self.graph_instance,1,0,1,2)
        graph_groupbox = qtw.QGroupBox()
        graph_groupbox.setLayout(graph_grid)
        graph_groupbox.setFont(self.custom_style2)
        graph_groupbox.setStyleSheet(box_style)
        return graph_groupbox

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
        restart_program()

    def color_coder(self, metrics_index, distribution=0, percent_max=0): # distribution[yellow, orange, red, purple]
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
            if value > last_value:
                return 'rgb(152,195,121)'
            elif value < last_value:
                return 'rgb(220,89,61)'

class GraphWidget(qtw.QWidget):
    'A widget to display a runnning graph of information'
    bad_color = qtg.QColor(255, 0, 0) #Red
    medium_color = qtg.QColor(255, 255, 0) #Yellow
    good_color = qtg.QColor(0, 255, 0) #Green

    def __init__(self, *args, data_width=100,
        minimum=0, maximum=100,
        warn_val=50, crit_val=75, scale=10, **kwargs):
        super().__init__(*args, **kwargs)

        self.minimum = minimum
        self.maximum = maximum
        self.warn_val = warn_val
        self.scale = scale
        self.crit_val = crit_val

        self.values = deque([self.minimum]* data_width, maxlen=data_width)
        self.setFixedWidth(data_width * scale)
    
    def add_value(self, value):
        '''
        This method begins by constraining our values between our min and max,
        and then appending it to the deque object
        '''
        value = max(value, self.minimum)
        value = min(value, self.maximum)
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
        pen.setColor(self.medium_color)
        painter.setPen(pen)
        painter.drawLine(0, warn_y, self.width(), warn_y)

        crit_y = self.val_to_y(self.crit_val)
        pen.setColor(self.bad_color)
        painter.setPen(pen)
        painter.drawLine(0, crit_y, self.width(), crit_y)

        gradient = qtg.QLinearGradient(qtc.QPointF(0, self.height()), qtc.QPointF(0, 0))
        gradient.setColorAt(0, self.good_color)
        gradient.setColorAt(self.warn_val/(self.maximum-self.minimum), self.medium_color)
        gradient.setColorAt(self.crit_val/(self.maximum-self.minimum), self.bad_color)
        
        brush = qtg.QBrush(gradient)
        painter.setBrush(brush)
        painter.setPen(qtc.Qt.NoPen)

        self.start_value = getattr(self, 'start_value', self.minimum)
        last_value = self.start_value
        self.start_value = self.values[0]

        for i, value in enumerate(self.values):
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

def map(x):
    result = ((x-1028)*(100-0)/(430-1028) + 0)
    return result

def launchcode(self):      #the universal 'show' protocol 
    self.setGeometry(1300, 70, 2530, 1980)
    self.setStyleSheet("background-color: rgb(40,44,52);")
    self.setWindowTitle('DRL Monitor')
    self.show()

def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    python = sys.executable
    os.execl(python, python, * sys.argv)

def main(connection):
    #if mode == 'Main_Window':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow(connection)   
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
metric_style = """
color: rgb(255,255,255);
"""

metric_style_random = """
border-color: rgb(152,195,121);
color: rgb(80,167,239);
border-width : 0px;
border-style:inset;
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