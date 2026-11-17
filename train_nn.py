
from biapy import BiaPy
import os
out_dir = "/nn_train_output"
os.makedirs(out_dir, exist_ok=True)

yaml_file = "YAML/semantic_config.yaml"

biapy = BiaPy(yaml_file, result_dir=out_dir, name="model_name", gpu="0", verbose=True)

biapy.run_job()