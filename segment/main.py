# coding=utf-8
import os
import tornado
from loguru import logger
from app.service import ServiceInstance
from infra.config.args import get_amg_kwargs
from infra.config.config import Log, Config, load_config_file
from infra.server.http import HttpServer
class App():
	def init(self):
		self.init_conf()
		self.init_logger(conf=self.conf.log)
		ServiceInstance.initializer(self.conf.models)	
		self.init_http_server()



	def init_conf(self):
		amg_kwargs = get_amg_kwargs()
		self.conf = load_config_file(amg_kwargs["config"])
		if self.conf is None:
			raise "load config failed"



	def init_logger(self,conf: Log):
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
		logger.info('tornado version',tornado.version_info)




	def init_http_server(self):
		self.http = HttpServer(self.conf.http)


	def run(self):
		self.http.run()


if __name__ == '__main__':
	app = App()
	app.init()
	logger.info("init finish")
	app.run()
