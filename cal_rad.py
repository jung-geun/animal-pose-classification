import math


def cal_rad(arr):
    rad = []

    a = math.atan2(arr["x"][0] - arr["x"][1], arr["y"][0] - arr["y"][1]) - math.atan2(
        arr["x"][1] - arr["x"][2], arr["y"][1] - arr["y"][2]
    )
    # print(a)
    rad.append(a)
    b = math.atan2(arr["x"][1] - arr["x"][2], arr["y"][1] - arr["y"][2]) - math.atan2(
        arr["x"][2] - arr["x"][3], arr["y"][2] - arr["y"][3]
    )
    rad.append(b)

    PI = math.pi

    deg = [(rad[0] * 180) / PI, (rad[1] * 180) / PI]
    # print(deg[0])

    return deg
