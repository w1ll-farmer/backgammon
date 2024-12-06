def log(board_before,roll,  move, board_after):
    myFile = open("./Data/log.txt",'a')
    myFile.write(f"{board_before}\t{roll}\t{move}\t{board_after}\n")
    myFile.close()