import os

AMPGEN_HOME = os.environ.get("AMPGEN_HOME", "/home/ubuntu/AMPGen_Product")
PROGEN_PATH = os.environ.get(
    "AMPGEN_PROGEN_PATH", os.path.join(AMPGEN_HOME, "progen2-xlarge")
)
DATA_PATH = os.environ.get("AMPGEN_DATA_PATH", os.path.join(AMPGEN_HOME, "data"))
MODEL_PATH = os.environ.get("AMPGEN_MODEL_PATH", os.path.join(AMPGEN_HOME, "models"))
OUTPUT_PATH = os.environ.get("AMPGEN_OUTPUT_PATH", os.path.join(AMPGEN_HOME, "output"))
