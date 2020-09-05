import torch
 
class config:
    USE_DOCKER = True
    use_meta = False
    mel_idx = 6
    out_dim = 9
    image_size = 640
    device = torch.device('cpu')
    enet_type = 'efficientnet_b7'
    kernel_type = "9c_b7_1e_640_ext_15ep"
    model_dir = "/home/abdurrehman/app/melanoma-best-single-model-no-meta-data" if USE_DOCKER else "E:/SIIM-ISIC-Melanoma-Classification/melanoma-best-single-model-no-meta-data"
    UPLOAD_FOLDER = "/home/abdurrehman/app/static" if USE_DOCKER else "E:/SIIM-ISIC-Melanoma-Classification/static"

    # DEBUG = True
    
    
    # DOCKER_UPLOAD_FOLDER = "/home/abdurrehman/app/static"  # in case of docker image
    # docker_model_dir = "/home/abdurrehman/app/melanoma-best-single-model-no-meta-data"
    
    # device = torch.device('cpu') if USE_DOCKER else torch.device('cuda') 
    
    # use_amp = False
    # predict_best = True
    # base_dir = 'D:/melanoma-datasets'
    # pred_dir = 'D:/siim-isic-melanoma-2020/melanoma-predictions'
    # trained_models_dir = '/melanoma-trained-models'
    
    # image_size = 640
    # data_dir = 'D:/melanoma-datasets/jpeg-melanoma-768x768'
    # data_dir2 = 'D:/melanoma-datasets/jpeg-isic2019-768x768'
    # kernel_type = '9c_b7_1e_640_ext_15ep_best_o_fold'
    # model_dir = 'D:/siim-isic-melanoma-2020/melanoma-best-single-model-no-meta-data'
    # enet_type = 'efficientnet-b7'
    
    # image_size = 256
    # data_dir = 'D:/melanoma-datasets/jpeg-melanoma-256x256'
    # data_dir2 = 'D:/melanoma-datasets/jpeg-isic2019-256x256'
    # kernel_type = 'effnetb3_256_meta_9a_ext_15epo_best_fold'
    # model_dir = 'D:/siim-isic-melanoma-2020/melanoma-trained-models'
    # enet_type = 'efficientnet-b3'
 
    # use_external = '_ext' in kernel_type
    # use_meta = False
    # # use_meta = 'meta' in kernel_type
    # mel_idx = 6