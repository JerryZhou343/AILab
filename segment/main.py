# coding=utf-8
import os
import sys
import argparse
from loguru import logger

from infra.config.args import get_amg_kwargs
from infra.config.config import Log, Config, load_config_file


def init_logger(conf: Log):
	logger.add(os.path.join(conf.output,"segment.log"),level=conf.level,rotation='1 week',retention=5,
	           compression='zip')
	logger.info(' segments/ ')
	logger.info('   /\     |     /\    ')
	logger.info(' /  \    |    /  \   ')
	logger.info('/    \   |   /    \  ')
	logger.info('      \  |  /      \ ')
	logger.info('       \ | /        \ ')
	logger.info('        \|/          \ ')
	logger.info('         v           v ')
	logger.info('  Starting segment...  ')
	logger.info('                       ')
	logger.info('     ~ Initiate voxels ~    ')
	logger.info('                       ')

def init_models():
	pass

if __name__ == '__main__':
	amg_kwargs = get_amg_kwargs()
	conf = load_config_file(amg_kwargs["config"])
	if conf is None:
		raise "load config failed"
	init_logger(conf.log)