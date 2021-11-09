import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

#win = pg.GraphicsWindow()
win = pg.plot()
#win = pg.PlotWidget(name='Action Space')
win.setWindowTitle('Action Space')
#w = np.array([0,4,8,12,16]) #spaced by 4
w = np.array([0,7,14,21,28]) #spaced by 7

x = np.arange(5)
data = [[30, 25, 50, 20, 20],
[40, 23, 51, 17, 40],
[35, 22, 45, 19, 50],
[13, 29, 55, 53, 20],
[26, 14, 25, 49, 35]]

base_color = 255

bg1 = pg.BarGraphItem(x=w,   height=data[0], width=1, brush=(25, base_color*0.2, 25))
bg2 = pg.BarGraphItem(x=w+1, height=data[1], width=1, brush=(25, base_color*0.4, 25))
bg3 = pg.BarGraphItem(x=w+2, height=data[2], width=1, brush=(25, base_color*0.6, 25))
bg4 = pg.BarGraphItem(x=w+3, height=data[3], width=1, brush=(25, base_color*0.8, 25))
bg5 = pg.BarGraphItem(x=w+4, height=data[4], width=1, brush=(25, base_color, 25))

labels = ["F", "P", "B", "L", "R"]

stringaxis = pg.AxisItem(orientation='bottom')
stringaxis.setTicks(labels)

win.addItem(bg1)
win.addItem(bg2)
win.addItem(bg3)
win.addItem(bg4)
win.addItem(bg5)
win.hideAxis('bottom')
win.setLabel(axis='bottom', text='X-axis')

# Final example shows how to handle mouse clicks:
class BarGraph(pg.BarGraphItem):
    def mouseClickEvent(self, event):
        print("clicked")



## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
