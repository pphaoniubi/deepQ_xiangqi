from attributes import *

def all_moves(piece_name, new_x, new_y, init_x, init_y):

    if piece_name.startswith("Black Chariot") or piece_name.startswith("Red Chariot"):
        # to implement
        if ((abs(new_x - init_x) <= 10 and new_y != init_y) or (new_x != init_x and abs(new_y - init_y) <= 10)):
            return True
        
    elif piece_name.startswith("Black Horse") or piece_name.startswith("Red Horse"):
        # to implement
            return True
    
    elif piece_name.startswith("Black Elephant") or piece_name.startswith("Red Elephant"):
        # to implement
            return True
    
    elif piece_name.startswith("Black Advisor") or piece_name.startswith("Red Advisor"):
        # to implement
            return True
    
    elif piece_name.startswith("Black General") or piece_name.startswith("Red General"):
        # to implement
            return True
    
    elif piece_name.startswith("Black Cannon") or piece_name.startswith("Red Cannon"):
        # to implement
            return True
    
    elif piece_name.startswith("Black Soldier") or piece_name.startswith("Red Soldier"):
        # to implement
            return False

    return False