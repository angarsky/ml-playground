from ultralytics import YOLO
from clearml import Task
import datetime

# Get current time.
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d__%H_%M')

# ClearML task management.
# See how to pass and manage params via ClearML: https://youtu.be/fnuAAxetwoU?t=638
task = Task.init(project_name='Pothole', task_name='My Model Training')

model_variant = 'yolov8n'
task.set_parameter('model_variant', model_variant)

args = dict(
    data='clearml/data/pothole.yaml',
    imgsz=640,
    epochs=2,
    batch=8,
    name='pothole_yolov8_' + formatted_time,
    cache=True
)
task.connect(args)

# Load the model.
model = YOLO(f"{model_variant}.pt")

# Training.
results = model.train(**args)

task.close()
