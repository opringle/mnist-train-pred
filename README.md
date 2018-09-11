# mnist-train-predict

Train and deploy mxnet gluon models for mnist prediction.

## Implentation Details

- Each convolutional layer is followed by batch normalization & relu activation
- Downsampling is achieved through 3 by 3 maxpooling filter with stride 2
- Dropout regularization can be applied between fully connected layers
- Softmax function at output layer
- Cross entropy loss with stochastic gradient descent used to optimize weights

## Instructions

### Venv

- Create a virtual env & install requirements. Eg:
    - `mkvirtualenv -a ./ -r requirements.txt -p python3 mnist`
- Train the model with default hyperparameters (time consuming without gpu): 
    - cpu: `(mnist) $ python train.py`
    - gpu: `(mnist) $ python train.py --gpus=1`
    - Model parameters will be saved to the `checkpoint` folder.
- Launch the model server locally: 
    - `(mnist) $ python server.py --params-file=./checkpoint/epoch9.params`
- [Download and unzip mnist jpegs](https://www.kaggle.com/scolianni/mnistasjpg)
- Send a request to the server
    - `curl -X POST -F "image=@<path to a jpg you downloaded>" http://localhost:8080/transform`
    
### Docker

- `$ Docker build --build-arg params_file=./checkpoint/epoch2.params --t= .`
- `$ Docker run`

## Notes

- Hyperparameters have not been optimized. This should be done with access to a gpu.
- Model server is cpu only.
- `train.py` has been factored to allow for easy integration with Amazon Sagemaker hyperparameter optimization.

## Branch plans

- add a dockerfile
- build docker image
- ensure model can be hosted from an image in your dockerhub