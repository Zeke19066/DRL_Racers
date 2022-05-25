import numpy as np



def image_scale_cal():
    #vals_y = [352,52,422,52]
    #vals_x = [859,104,1046,104]
    vals_y = [1360-52-1] 
    vals_x = [202-104-1]
    val_out = []

    #1819x1421px
    #to126x126
    y_max, x_max = 1421, 1819

    out_y, out_x = 925, 873

    for val in vals_y:
        val_temp = np.interp(val,[0, y_max],[0,out_y]) #like c map function:(input val, [inrange_min,inrange_max],[outrange_min,outrange_max]
        val_out.append(round(val_temp))
    for val in vals_x:
        val_temp = np.interp(val,[0, x_max],[0,out_x]) #like c map function:(input val, [inrange_min,inrange_max],[outrange_min,outrange_max]
        val_out.append(round(val_temp))
    
    print(vals_y,vals_x)
    print(val_out)

def image_crop_calc():

    #vals_y = [352,52,422,52]
    #vals_x = [859,104,1046,104]
    vals_y = [948-52,952-52] 
    vals_x = [888-104,892-104, 1138-104,1142-104]
    val_out = []

    #1715x1369px
    #to 126x126
    y_max, x_max = 1421, 1819

    out_y, out_x = 925, 873
    for val in vals_y:
        val_temp = val-y_max
        val_out.append(round(val_temp))
    for val in vals_x:
        val_temp = val-x_max
        val_out.append(round(val_temp))
    print(val_out)

image_scale_cal()
