from predict import predict_image

if __name__ == "__main__":
    pred = predict_image(image_path="E:/data/jpeg-melanoma-768x768/ISIC_0208233.jpg")
    print(pred)     