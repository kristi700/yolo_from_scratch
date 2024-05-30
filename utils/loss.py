import torch

from torch import nn

class YoloLoss():
    def __init__(self, input_size, anchors, label_smoothing=0.0):
        self.stride = 32
        self.anchors = anchors
        self.num_attributes = 1 + 4 + 1 # obj, xywh, label (no need for onehot as this is only used for loss calc purposes)
        self.iou_threshold = 0.5 # 
        self.obj_loss = nn.MSELoss()
        self.box_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._set_grid_yx(input_size)

    def _set_grid_yx(self, input_size):
        self.grid_size = input_size // self.stride
        # TODO do the rest when needed

    def __call__(self, predicitions, labels):
        # build targets -> 
            # for each batch, and for each box in each cell we need to get the best anchors
            # Assign the ground truth values (coordinates, objectness score, and class label) to the appropriate positions in the target tensors.

        targets = self._build_targets(labels)

        #Extract the predicted bounding box coordinates, objectness scores, and class probabilities from the predictions tensor

        #Calculate the box loss using the mean squared error between the predicted and target bounding box coordinates.
        #Calculate the objectness loss using the mean squared error between the predicted and target objectness scores.
        #Calculate the class loss using the cross-entropy loss between the predicted and target class probabilities.

        #Sum the individual losses (box loss, objectness loss, and class loss) to get the total loss.
        

    def _build_targets(self, labels):
        targets = torch.stack([self._build_target(label) for label in labels])
        return targets

    def _build_target(self, label):
        # target shape -> grid*grid*bboxnum*attributesnum
        target = torch.zeros(size=(self.grid_size, self.grid_size, len(self.anchors), self.num_attributes))
        for item in label:
            #get classid
            class_id = item[0].long()
            # claculate the grid that's relevant
            grid_ij = ((item[1] * self.grid_size).long(), (item[2] * self.grid_size).long())
            # iou with anchors -> get the best anchor
            ious_target_with_anchor = self.calculate_iou_target_with_anchors(item[3:5])
            best_iou_id = torch.argmax(ious_target_with_anchor)
            # x,y in respect to current grid
            target_x = (item[1] * self.grid_size) - grid_ij[0]
            target_y = (item[2] * self.grid_size) - grid_ij[1]

            # w,h we need to better the anchor
            # we will stick to the log calculations for now as in yolov2 paper, but will change it to n*sigmoid later for better stability
            target_w = torch.log(item[3] / self.anchors[best_iou_id][0])
            target_h = torch.log(item[4] / self.anchors[best_iou_id][1])

            # fill the target grid ij anchorid with calculated values
            # target last dim is - objectness, xywh, classid
            target[grid_ij[0], grid_ij[1], best_iou_id, 1:5] = torch.tensor([target_x, target_y, target_w, target_h])
            target[grid_ij[0], grid_ij[1], best_iou_id, 5] = class_id

            # set all anchors objectness to -1, which is higher than iou, leave the rest as 0
            for id, iou in enumerate(ious_target_with_anchor):
                if id == best_iou_id:
                    target[grid_ij[0], grid_ij[1], id, 0] = 1.0
                elif iou > self.iou_threshold:
                    target[grid_ij[0], grid_ij[1], id, 0] = -1.0
            return target

    
    def calculate_iou_target_with_anchors(self, target_wh):
        t_w, t_h = target_wh
        anchor_w, anchor_h = self.anchors.t()

        inter_w = torch.min(t_w, anchor_w)
        inter_h = torch.min(t_h, anchor_h)

        inter_area = inter_w * inter_h

        target_area = t_w * t_h
        anchor_areas = anchor_w * anchor_h

        union = target_area + anchor_areas - inter_area

        return inter_area/union





