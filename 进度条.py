#author: wmingzhu
#date: 2024/08/14
import time
from colorama import Fore, Back,Style


def print_progress(current,total,start):
    """
     打印任务进度
    :param current: 当前step
    :param total:   总的step数
    :return: None
    """
    percent = str(round((current / total) * 100, 2))
    int_part, float_part = percent.split(".")

    if len(int_part) < 3:
        int_part = (3 - len(int_part)) * "0" + int_part
    if len(float_part) < 2:
        float_part = float_part + (2 - len(float_part)) * "0"

    formated_percent = int_part + "." + float_part

    print("\r" + Back.LIGHTCYAN_EX + Fore.LIGHTCYAN_EX + "{}".format(
        int(int_part) * "|") + Style.RESET_ALL + Back.BLACK + Fore.BLACK + "{}".format(
        (100 - int(int_part)) * "|") + Style.RESET_ALL + " {}%".format(formated_percent), end="")
    print(" 总耗时:{}s".format(round(time.perf_counter() - start, 1)), end="")

if __name__ == "__main__":
    #测试
    start = time.perf_counter()
    for i in range(1,1001):
        print_progress(i, 1000,start)
        time.sleep(0.01)