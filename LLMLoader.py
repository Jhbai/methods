from gemma3 import Gemma3Object
from qwen3 import Qwen3Object

def LoadModel(path):
    if "gemma3" in path.lower():
          model = Gemma3Object(path)
    else:
        model = Qwen3Object(path)
    return model
