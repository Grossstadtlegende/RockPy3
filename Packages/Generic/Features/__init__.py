import os
abs_module= 'RockPy3'+ os.path.dirname(__file__).split('RockPy3')[-1].replace('/', '.')
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    __import__('.'.join([abs_module, module[:-3]]), locals(), globals())
del module