#coding=utf-8
import argparse


argument_parse = argparse.ArgumentParser(
    description=(
        '''segment picture'''
    )
)

argument_parse.add_argument(
    '''--config''',
    type=str,
    required=False,
    help="path to config file",
    default="./conf/config.yaml"
)

def get_amg_kwargs()->dict:
    args =argument_parse.parse_args()
    amg_kwargs = {
        "config":args.config
    }

    return amg_kwargs