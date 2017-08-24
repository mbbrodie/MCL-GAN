import sys
f = open(sys.argv[1], 'r')
X = set()
out = open(sys.argv[2], 'w')
for l in f.readlines():
    if l[0:17] not in X:
        X.add(l[0:17])
        out.write(l)    
print X
f.close()
out.close()
