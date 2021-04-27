# merge 10 classes into 4 classes
class_names = [
    'vehicle',  # car, truck, bus, trailer, cv
    'pedestrian',  # pedestrian
    'bike',  # motorcycle, bicycle
    'traffic_boundary'  # traffic_cone, barrier
    # background
]

dataset_type = 'MMDAMergeCatDataset'
train_pipeline, test_pipeline
