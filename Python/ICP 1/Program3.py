import random
x=random.randint(0,9)
print(x)
while True:
    GuessNum = int(input("Guess the Number:"))

    if GuessNum>x:
        print("Number is Greater")
    elif GuessNum<x:
        print("Number is Less")
    else:
        print("Number is PERFECT")
        break