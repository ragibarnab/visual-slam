from collections import OrderedDict


od = OrderedDict()

od['a'] = 1
od['b'] = 2
od['c'] = 3

x = {(1, 2): 3, (4, 5): 6}

for (a, b), v in x.items():
    print(a, b)