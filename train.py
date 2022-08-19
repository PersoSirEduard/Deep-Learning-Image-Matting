import enum
import torch
from tqdm import tqdm
from model import UNet
from dataset import get_loaders
from utils import check_accuracy

learning_rate = 0.0001
num_epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 10
n_workers = 2
image_size = 256
pin_memory = True
img_dir = "E:\Full_Body_Segmentation_Dataset\images\*.png"
mask_dir = "E:\Full_Body_Segmentation_Dataset\masks\*.png"

def train(loader, model, optimizer, loss_fn, scaler):

    loop = tqdm(loader)

    for id, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        with torch.cuda.amp.autocast():
            preds = model(data)
            loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

if __name__ == '__main__':
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    # model.load_state_dict(torch.load("backup/model_24.pth"))
    loss_fn = torch.nn.BCEWithLogitsLoss() # for multi-class classification use torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = get_loaders(img_dir, mask_dir, batch_size, n_workers, pin_memory)

    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, loss_fn, scaler)

        torch.save(model.state_dict(), f"model_{epoch}.pth")
        accuracy, dice = check_accuracy(val_loader, model, device=device)
        print(f"Epoch: {epoch}\tAccuracy: {accuracy}\tDice Score: {dice}")
        
