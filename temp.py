d = {1.0:2,2:3,3:4}
def f(di):
    di[100] = 100

f(d)
print(1.0 in d.keys())
d2 = {1000:1000}
print({**d2,**d})