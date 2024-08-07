# On local machine
import fiftyone as fo

dataset = fo.Dataset("my-dataset")

session = fo.launch_app(dataset)  # (optional) port=XXXX