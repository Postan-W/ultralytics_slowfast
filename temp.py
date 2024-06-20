import numpy as np

a = np.array([[22,1,33,3,999,13,15,47],[22,1,33,3,999,13,15,47],[22,1,33,3,999,13,15,47]])
result = []
for i in a:
    result.append(i[np.argsort(-i)].tolist())

print(result)
s = ""
for i in range(4):
    s += " " + str(i)
print(s)