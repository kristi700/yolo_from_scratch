import torch

from alive_progress import alive_bar

def validate(dataloader, model, criterion, device):
    model.eval() 
    val_loss = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        with alive_bar(len(dataloader)) as bar:
            for minibatch in dataloader:
                images, labels = minibatch[0], minibatch[1]
                images = images.to(device)

                outputs = model(images).to(device)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                bar()

    avg_val_loss = val_loss / num_batches
    return avg_val_loss