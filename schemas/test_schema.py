import jsonschema
import json
from luxonis_train.utils.config import Config
import yaml

cfg = Config("../configs/simple_det.yaml")
data = cfg.get_data()

# with open("../configs/simple_det.yaml") as f:
#     data = yaml.load(f, Loader=yaml.SafeLoader)

with open("config_schema_full.json") as f:
    schema = json.load(f)

try:
    # Validate the data against the schema
    jsonschema.validate(data, schema)
    print("Validation successful!")
except jsonschema.exceptions.ValidationError as e:
    print(f"Validation failed: {e}")