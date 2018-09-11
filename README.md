# mnist-train-pred
Train and deploy mxnet models for mnist prediction

## Additions 

- Dropout regularization in fully connected layers
- Batch normalization between conv layers

## Instructions

- Create a virtual env. Eg:
    - `mkvirtualenv -a ./ -r requirements.txt -p python3 mnist`
- Train the model with default hyperparameters (time consuming without gpu): 
    - `$ python train.py`
    - Model parameters will be saved to the `checkpoint` folder.
- Launch the model server locally: 
    - `$ python server.py --params-file=./checkpoint/epoch10.params`
- [Download and unzip mnist jpegs](https://www.kaggle.com/scolianni/mnistasjpg)
- Send a request to the server
    - `curl -X POST -F "image=@<path to a jpg>" http://localhost:8080/transform`

## ToDo

- [ ] Softmax output must sum to 1