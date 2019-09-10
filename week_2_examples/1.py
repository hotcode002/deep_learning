user_floor = 3
current_floor = 6
 
difference = user_floor - current_floor
 
if difference < 0 : 
    current_floor = user_floor
    print ( " Move down ")
     
elif difference > 0 : 
    current_floor = user_floor
    print ( " Move up ")