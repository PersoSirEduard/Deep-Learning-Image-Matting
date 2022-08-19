import torch
from model import UNet
from dataset import get_loaders
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import cv2


img_dir = "E:\Supervise.ly_Dataset\images\*.png"
mask_dir = "E:\Supervise.ly_Dataset\masks\*.png"
model_checkpoint = "my_model3.pth"

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_checkpoint))

    _, valid_dl = get_loaders(img_dir, mask_dir, batch_size=6, n_workers=4, pin_memory=True)

    data = next(iter(valid_dl))
    img = data[0].to(device)
    mask = data[1].unsqueeze(1)[1].numpy()
    preds = torch.sigmoid(model(img))
    preds = (preds > 0.5).float()
    preds = preds.cpu().numpy()[1]
    

    fig, ax = plt.subplots(1, 4)

    default_img = img.cpu()[1].permute(1, 2, 0).numpy()

    ax[0].imshow(default_img)
    ax[0].set_title("Input Image")

    ax[1].imshow(resize(preds[0], (256, 256)), cmap="gray")
    ax[1].set_title("Predicted Mask")

    ax[2].imshow(resize(mask[0], (256, 256)), cmap="gray")
    ax[2].set_title("Actual Mask")

    # print(cv2.cvtColor(resize(preds[0], (256, 256)), cv2.COLOR_GRAY2RGB).shape)

    mask = resize(preds[0], (256, 256)) > 0.5 
    mask = np.expand_dims(mask, axis=2)
    # Boolean to int 8
    mask = mask.astype(np.uint8)

    ax[3].imshow(cv2.bitwise_and(default_img, default_img, mask=mask))
    ax[3].set_title("Applied predicted Mask")

    plt.show()



