strings=["programming","python","deepLearning"]
leng=[len(strings[0]),len(strings[1]),len(strings[2])]
print(strings)
print(leng)
Matrix = [[0 for x in range(2)] for y in range(len(strings))]
for u in range(len(strings)):
    for v in range(0,1):
        Matrix[u][v] = leng[u]
        Matrix[u][1] = strings[u]
print(Matrix)

print(max(Matrix))
