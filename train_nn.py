
from biapy import BiaPy
import os
out_dir = "/facility/imganfac/neurogenomics/taverna/scordato_cilia_v2/model_training_v2"
os.makedirs(out_dir, exist_ok=True)

yaml_file = "/facility/imganfac/neurogenomics/taverna/scordato_cilia_v2/YAML/semantic_config.yaml"

biapy = BiaPy(yaml_file, result_dir=out_dir, name="model_name", gpu="0", verbose=True)

biapy.run_job()