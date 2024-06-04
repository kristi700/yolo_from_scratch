import torch

from utils.datasets import VOCDataset
from utils.model.yolo import YoloModel
from utils.loss import YoloLoss

from alive_progress import alive_bar
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW

# TODO - create val set and val it on the fly
# TODO - save the model trained (ckpt the best one yet based on val acc)
# TODO - create test notebook where one can try the model

def train(dataloader, model, criterion, optimizer, device): # , model, criterion, optimizer, scaler
    optimizer.zero_grad()

    with alive_bar(len(dataloader)) as bar:
        for i, minibatch in enumerate(dataloader):
            # TODO add tqdm for epochs
            # TODO add nubmer of epoch running now
            images, labels = minibatch[0], minibatch[1]
            predictions = model(images).to(device)
            loss = criterion(predictions=predictions, labels=labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bar()
    return loss


def main():
    #HARDCODED for now
    csv_file = 'PASCAL/100examples.csv'
    img_dir = 'PASCAL/images'
    label_dir = 'PASCAL/labels'
    epochs = 10
    batch_size=8
    device = torch.device("cpu")
    #######
    # TODO -> add augmentations later on
    transfrom = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])
    training_data = VOCDataset(csv_file, img_dir, label_dir, transform= transfrom)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=training_data.collate_fn)
    # TODO THIS MIGHT NOT BE OPTIMAL FOR VOC -> needs k means for this later when training "works"
    anchors = [[0.47070834, 0.7668643 ],
               [0.6636637,  0.274     ],
               [0.875,      0.61066663],
               [0.8605263,  0.8736842 ],
               [0.283375,   0.5775    ]]
    
    model = YoloModel((416, 416), num_classes=20, anchors=anchors).to(device)
    # TODO could try to use different lr for backend and head
    # TODO could try to not weight dexay bias params
    # https://pytorch.org/docs/stable/optim.html
    optimizer = AdamW(params=model.parameters(), lr= 0.001)
    criterion = YoloLoss(input_size=model.input_size[0], batch_size=batch_size, anchors=model.anchors, device=device, label_smoothing=0.1)
    # image test - cv2.imwrite("asd.jpg", np.transpose(torch.Tensor.numpy(training_data[2][0]), (1,2,0))*255)
    for epoch in range(0, 10):
        print(f"Epoch: {epoch}\n")
        current_loss = train(dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer, device=device)
        print(f"train loss: {current_loss}\n")


if __name__ == "__main__":
    main()