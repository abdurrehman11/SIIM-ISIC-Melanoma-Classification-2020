# SIIM-ISIC-Melanoma-Classification-2020
This repo will contain an end-to-end deep learning application.

![image](https://github.com/abdurrehman11/SIIM-ISIC-Melanoma-Classification-2020/assets/24878579/b0c343e2-0e37-49ff-998f-a6e98bc3e6fa)


# Data
- You can download the datasets from the following links: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164092 and https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164910.
- You can download the pre-trained 5-fold model for inference from here: https://www.kaggle.com/boliu0/melanoma-best-single-model-no-meta-data

# SOFTWARE (python packages are detailed separately in requirements.txt)
- Python 3.8
- Docker
- Pytorch (make sure to specify the version in `requirements.txt` compatible with your CUDA version)


# Setup & Installation
- Clone the repo on your local machine and go to the root directory of the project
- Install virtualenv using `pip install virtualenv`
- Create virtual env using `python -m venv env` 
- Activate virtual env using `.\env\Scripts\activate` (for Windows)
- Upgrade pip using `pip install --upgrade pip`
- Install dependencies using `pip install -r requirements.txt`

# Model Training, Evaluation, Prediction & Ensembling
Execute the below commands by passing your dataset path to the `data-dir` parameter to start training, evaluation, prediction, and ensembling.

After training, models will be saved in `./weights/` Tranning logs will be saved in `./logs/`

`python train.py --kernel-type b3_256_256_meta_ext_15ep --data-dir /data/ --data-folder 256 --image-size 256 --enet-type efficientnet_b3 --use-meta`

Evaluation results will be printed out and saved to `./logs/` Out-of-folds prediction results will be saved to `./oofs/`

`python evaluate.py --kernel-type b3_256_256_meta_ext_15ep --data-dir /data/ --data-folder 256 --image-size 256 --enet-type efficientnet_b3 --use-meta`

Each model submission file will be saved to `./subs/`

`python predict.py --kernel-type b3_256_256_meta_ext_15ep --data-dir E:/data/ --data-folder 256 --image-size 256 --enet-type efficientnet_b3 --use-meta`

Ensemble of every single model's submission files (from the previous step) into the final submission file will be save in the root directory as `final_sub1.csv`

`python ensemble.py`

# Web App Setup & Demo
- Build Docker image using `docker build -f Dockerfile -t melanoma_api:v0 .`
- Run Docker image using `docker run -p 5000:5000 -ti melanoma_api:v0 python api.py`
- Test the app by entering the following URL in the browser: `http://localhost:5000` and predict your image and enjoy!
