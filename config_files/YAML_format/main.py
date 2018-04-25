import yaml

with open("config.yml", 'r') as yml_file:
    cfg = yaml.load(yml_file)

for section in cfg:
    print(cfg[section])