from attributes import *

def all_moves(piece_name, new_x, new_y, init_x, init_y):

    image, rect = pieces[f"Black Chariot 0"]

    if piece_name.startswith("Black Chariot") or piece_name.startswith("Red Chariot"):
        # to implement
        if ((new_x == init_x and new_y != init_y) or (new_x != init_x and new_y == init_y)):
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
    
    elif (piece_name == "Black General" or piece_name == "Red General"):
        # to implement
            return True
    
    elif (piece_name == "Black Cannon" or piece_name == "Red Cannon"):
        # to implement
            return True
    
    elif (piece_name == "Black Soldier" or piece_name == "Red Soldier"):
        # to implement
            return True

    return False