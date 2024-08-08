d = {1.0:2,2:3,3:4}
def f(di):
    di[100] = 100

f(d)
print(1.0 in d.keys())
d2 = {1000:1000}
print({**d2,**d})

a = [1,2,3,4,5,6,7]
print(a[len(a)-5:])