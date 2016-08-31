from utils import param_init


a = param_init().uniform([10, 10])

print a.get_value()