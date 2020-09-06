# SIIM-ISIC-Melanoma-Classification-2020
This repo will contain end to end deep learning application. Deep learning code used here from Great grandmasters of Kaggle Melanoma2020 competition winner. You can find the winner code 
here : https://www.kaggle.com/haqishen/1st-place-soluiton-best-model-infer/data?

# Data
- You can download the datasets from the following links: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164092 and https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164910.
- You can download the pre-trained 5-fold model for inference from here: https://www.kaggle.com/boliu0/melanoma-best-single-model-no-meta-data

# SOFTWARE (python packages are detailed separately in requirements.txt)
- Python 3.8
- Docker
- Pytorch (make sure to specify the version in `requirements.txt` compatible with your CUDA version)


# Setup & Installation
- Clone the repo on your local machine and goto root directory of project
- Install virtualenv using `pip install virtualenv`
- Create virtual env using `python -m venv env` 
- Activate virtual env using `.\env\Scripts\activate` (for Windows)
- Upgrade pip using `pip install --upgrade pip`
- Install dependencies using `pip install -r requirements.txt`

# Model Training, Evaluation and Prediction
Execute the below commands by passing your dataset path to `data-dir` parameter to start training, evaluation and prediction.

After training, models will be saved in `./weights/` Tranning logs will be saved in `./logs/`

`python train.py --kernel-type b3_256_256_meta_ext_15ep --data-dir /data/ --data-folder 256 --image-size 256 --enet-type efficientnet_b3 --use-meta`

Evaluation results will be printed out and saved to `./logs/` Out-of-folds prediction results will be saved to `./oofs/`

`python evaluate.py --kernel-type b3_256_256_meta_ext_15ep --data-dir /data/ --data-folder 256 --image-size 256 --enet-type efficientnet_b3 --use-meta`

Each models submission file will be saved to `./subs/`

`python predict.py --kernel-type b3_256_256_meta_ext_15ep --data-dir E:/data/ --data-folder 256 --image-size 256 --enet-type efficientnet_b3 --use-meta`

# Web App Setup & Demo
- Build Docker image using `docker build -f Dockerfile -t melanoma_api:v0 .`
- Run Docker image using `docker run -p 5000:5000 -ti melanoma_api:v0 python api.py`
- Test app by hit the following url in browser: `http://localhost:5000` and predict your image and enjoy!
