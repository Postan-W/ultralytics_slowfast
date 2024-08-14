#author: wmingzhu
#date: 2024/08/12
import time
from colorama import Fore, Back,Style
start = time.perf_counter()
for i in range(1,1001):
    time.sleep(0.001)#其实调用time.sleep本身也消耗时间，不只是睡眠的时间，通过增加调用次数可以发现
    percent = str(round((i/1000)*100,2))
    int_part,float_part = percent.split(".")
    if len(int_part) < 3:
        int_part = (3-len(int_part))*"0" + int_part
    if len(float_part) < 2:
        float_part = float_part + (2-len(float_part))*"0"

    formated_percent = int_part + "." + float_part
    left_fill = int(int_part)
    right_fill = 100-int(int_part)

    print("\r"+ Back.LIGHTCYAN_EX + Fore.LIGHTCYAN_EX + "{}".format(int(int_part)*"|") + Style.RESET_ALL + Back.BLACK + Fore.BLACK + "{}".format((100-int(int_part))*"|") + Style.RESET_ALL + " {}%".format(formated_percent),end="")
    print(" 总耗时:{}s".format(round(time.perf_counter()-start,1)),end="")



