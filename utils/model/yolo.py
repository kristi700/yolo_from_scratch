import torch

from utils.model.backbone import Darknet19

from torch import nn

class YoloModel(nn.Module):
    def __init__(self, input_size, num_classes, anchors):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors)
        self.num_boxes = 5
        self.num_attributes = 1 + 4 + num_classes # obj, x, y, w, h

        # TODO -> 'neck' -> concat 26x26 and 13x13 feature maps
        self.backbone = Darknet19()
        self.head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, self.num_boxes * self.num_attributes, kernel_size=1))
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        batch_size, _, grid_size, _ = x.shape
        # now -> batch , numbox * numattr, grid, grid
        # want -> batch, numbox, grid, grid, num attr
        x = x.view(batch_size, self.num_boxes, self.num_attributes, grid_size, grid_size)
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        # box center points needs sigmoid!
        # objectness needs sigmoid as well

        pred_obj = torch.sigmoid(x[..., [0]])
        pred_box_txty = torch.sigmoid(x[..., 1:3])
        pred_box_twth = x[..., 3:5]
        pred_cls = x[..., 5:] # TODO - weird for me how we dont softmax class labels - could help
        
        return torch.cat((pred_obj, pred_box_txty, pred_box_twth, pred_cls), dim=-1)
        
