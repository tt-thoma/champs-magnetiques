# -*- encoding: utf-8 -*-

debug = True


def print_debug(*args):
    global debug
    if not debug:
        return
    for i in args:
        print("-", i, end="")
    print()
