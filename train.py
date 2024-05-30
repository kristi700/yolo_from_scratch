import torch

from utils.datasets import VOCDataset
from utils.model.yolo import YoloModel
from utils.loss import YoloLoss

from torchvision import transforms
from torch.utils.data import DataLoader

def train(dataloader, model, criterion, device): # , model, criterion, optimizer, scaler
    for i, minibatch in enumerate(dataloader):
        images, labels = minibatch[0], minibatch[1]
        predicitions = model(images).to(device)
        loss = criterion(predicitions=predicitions, labels=labels)

def main():
    #HARDCODED for now
    csv_file = 'PASCAL/100examples.csv'
    img_dir = 'PASCAL/images'
    label_dir = 'PASCAL/labels'
    epochs = 10
    device = torch.device("cpu")
    #######
    # TODO -> add augmentations later on
    transfrom = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])
    training_data = VOCDataset(csv_file, img_dir, label_dir, transform= transfrom)
    train_loader = DataLoader(training_data, batch_size=8, shuffle=True, collate_fn=training_data.collate_fn)

    # THIS MIGHT NOT BE OPTIMAL FOR VOC -> needs k means for this later when training "works"
    anchors = [[0.47070834, 0.7668643 ],
               [0.6636637,  0.274     ],
               [0.875,      0.61066663],
               [0.8605263,  0.8736842 ],
               [0.283375,   0.5775    ]]
    
    model = YoloModel((416, 416), num_classes=20, anchors=anchors).to(device)
    criterion = YoloLoss(input_size=model.input_size[0], anchors=model.anchors, label_smoothing=0.1)
    # image test - cv2.imwrite("asd.jpg", np.transpose(torch.Tensor.numpy(training_data[2][0]), (1,2,0))*255)
    for epoch in range(0, 10):
        current_loss = train(dataloader=train_loader, model=model, criterion=criterion, device=device)


if __name__ == "__main__":
    main()