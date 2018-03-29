from enum import Enum


class State(Enum):
    power_off = 1  # 关机
    not_exist = 2  # 空号
    overdue = 3  # 欠费
    out_of_service = 4  # 停机
    other = 0  # 其他
