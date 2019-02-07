fileName = input("Please enter the input file location ")
infile = open(fileName,'r')
line = infile.readline()
output= open("output.txt","w")
while line != "":
    for x in line.split("\n"):
        print("%s, %d\n"%(line.rstrip("\n"),len(line.rstrip("\n"))))
        output.write("%s, %d\n"%(line.rstrip("\n"),len(line.rstrip("\n"))))
        line = infile.readline()
output.close()