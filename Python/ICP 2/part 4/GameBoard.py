heightInput = int(input("Enter the height of the board: "))
widthInput = int(input("Enter the width of the board: "))
def board_draw(height, width):
    board = ""
    for count in range(height):
        board = board+" ---" * (width) + "\n" + "|   " * (int(width)) + "|\n"
    board = board+" ---" * (int(width)) + "\n"
    return board
print(board_draw(heightInput, widthInput))
