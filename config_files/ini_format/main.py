import configparser
from configparser import ConfigParser

# instantiate
cfg = ConfigParser()

# parse existing file
cfg.read('config.ini')

for section in cfg.sections():
	print(cfg.items(section))


