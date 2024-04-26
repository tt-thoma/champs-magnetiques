# -*- encoding: utf-8 -*-

debug = False

def print_debug(*args):
    global debug
    if not debug:
        return
    print(args)
