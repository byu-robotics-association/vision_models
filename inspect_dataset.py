import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

project = rf.workspace("mhseals").project("buoys-4naae")

print("Dataset classes:")
print(project.classes)