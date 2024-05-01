# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib.colors as mcolors
debug = False



def print_debug(*args):
    global debug
    if not debug:
        return
    for i in args:
        print("-", i, end="")
    print()
    
    
    
