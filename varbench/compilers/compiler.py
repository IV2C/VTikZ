from abc import ABC, abstractmethod

class Compiler(ABC):
  def __init__(self):
    pass
  
  @abstractmethod
  def compile(self, input, output):
    pass
  
  @abstractmethod
  def compile_from_string(self, input_string, output):
    pass