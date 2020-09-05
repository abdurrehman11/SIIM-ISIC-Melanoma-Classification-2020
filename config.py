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
