#coding=utf-8

import os

def check_file(file_name:str)->bool:
    if not os.path.exists(file_name):
        return False
    if  not os.path.isfile(file_name):
        return False
    return True

