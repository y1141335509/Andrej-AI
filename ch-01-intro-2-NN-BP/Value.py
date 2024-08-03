import numpy as np


class Value:
  def __init__(self, data) -> None:
    self.data = data

  def __repr__(self) -> str:
    """
      data wrapper
      用来覆写 print() 方法的输出结果
    """
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    """
      用来覆写 + 的输出结果
    """
    out = Value(self.data + other.data)
    return out
  
  def __mul__(self, other):
    """
      用来覆写 * 的输出结果
    """
    out = Value(self.data * other.data)
    return out
