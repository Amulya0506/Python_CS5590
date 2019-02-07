fileName = input("Please enter the input file location ")
infile = open(fileName,'r')
line = infile.readline()
while line != "":
    length = 0
    for x in line.split(" "):
        length = length+1
    print("%s, %d\n" % (line.rstrip("\n"), length))
    line = infile.readline()