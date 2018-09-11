# mnist-train-predict

Train and deploy mxnet gluon models for mnist prediction.

## Additions 

- Dropout regularization in fully connected layers
- Batch normalization between conv layers

## Instructions

- Create a virtual env & install requirements. Eg:
    - `mkvirtualenv -a ./ -r requirements.txt -p python3 mnist`
- Train the model with default hyperparameters (time consuming without gpu): 
    - `(mnist) $ python train.py`
    - Model parameters will be saved to the `checkpoint` folder.
- Launch the model server locally: 
    - `(mnist) $ python server.py --params-file=./checkpoint/epoch9.params`
- [Download and unzip mnist jpegs](https://www.kaggle.com/scolianni/mnistasjpg)
- Send a request to the server
    - `curl -X POST -F "image=@<path to a jpg you downloaded>" http://localhost:8080/transform`

## Notes

- Hyperparameters have not been optimized. This should be done with access to a gpu.
- Model server operates on cpu.
- `train.py` has been factored to allow for easy integration with Amazon Sagemaker hyperparameter optimization.