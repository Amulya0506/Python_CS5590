word=input("Enter a string: ")
vowels=set('aeiou')
print(vowels)
count= 0
for char in word:
    if char in vowels:
        count +=1
print("Number of vowels: %d"%(count))

