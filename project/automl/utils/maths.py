

def nearest_multiple(Y, W):
    lower = (Y // W) * W
    upper = lower + W
    
    if abs(Y - lower) <= abs(upper - Y):
        return lower
    else:
        return upper
    

def nearest_highest_multiple(Y, W):
    lower = (Y // W) * W
    upper = lower + W
    
    return upper