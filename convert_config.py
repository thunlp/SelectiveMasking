import json
import sys
import os

config_path = sys.argv[1]

with open(config_path) as f:
    config = json.load(f)

os.makedirs("config/", exist_ok=True)

f = open("config/bash_config.sh", "w")

def set_env(d, prefix):
    for name in d:
        if isinstance(d[name], dict):
            set_env(d[name], prefix + "_" + name)
        else:
            # print("export {}={}".format((prefix + "_" + name).upper(), d[name]))
            # os.system("export {}={}".format((prefix + "_" + name).upper(), d[name]))
            # os.system("echo ${}".format((prefix + "_" + name).upper()))
            f.write("{}={}\n".format((prefix + "_" + name).upper(), d[name]))

set_env(config, "E")

f.close()
