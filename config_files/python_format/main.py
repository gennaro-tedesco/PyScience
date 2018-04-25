import config as cfg

for key in cfg.config.keys():
    print("{}: {}".format(key, cfg.config[key]))