# anomalydetection
Finds out anomalies with taking differences from the images which can be generated with VAE defined in this repo.

### Quickstart
To start inference for predefined image datasets (which is not included), just type:
```
python3 inference.py
```
Out of the box, in many cases we need to execute the inference program, so type:
```
python3 training.py
```
to build VAE models with specified datasets.

### Configure parameters
Open `param.py`, try changing some params about datasets and a learning period.
Parameters for the model is separated in `training.py`.