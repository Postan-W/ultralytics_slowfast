import time
for i in range(10000):
    time.sleep(0.001)
    print("\r$$${}%$$$".format(round((i/10000)*100,2)),end="")