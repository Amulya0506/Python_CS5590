import sys
# Taking list of numbers from user
print("Please enter numbers to find sum zero of three numbers :")
numbers = sys.stdin.readline().strip('\n').split(" ")
print(numbers)
triplets = []
# Loop the list to find three numbers so that it sum is zero
for num1 in range(0,len(numbers)):
    for num2 in range(num1+1,len(numbers)):
        for num3 in range(num2+1,len(numbers)):
            if int(numbers[num1])+int(numbers[num2])+int(numbers[num3]) == 0:
                    triplet = (int(numbers[num1]), int(numbers[num2]), int(numbers[num3]))
                    # appending each triplet to the 'triplets' list
                    triplets.append(triplet)

print("Number of triplets: %d"%(len(triplets)))
print("Triplets are: ")
print(triplets)