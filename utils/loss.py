import torch

from torch import nn

class YoloLoss():
    def __init__(self, input_size, batch_size, anchors, device, label_smoothing=0.0):
        self.stride = 32
        self.anchors = anchors
        self.num_attributes = 1 + 4 + 1 # obj, xywh, label (no need for onehot as this is only used for loss calc purposes)
        self.iou_threshold = 0.5
        self.device = device
        self.batch_size = batch_size
        self.obj_loss = nn.MSELoss()
        self.box_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._set_grid_yx(input_size)

    def _set_grid_yx(self, input_size):
        self.grid_size = input_size // self.stride
        grid_y, grid_x = torch.meshgrid((torch.arange(self.grid_size), torch.arange(self.grid_size)), indexing="ij")
        self.grid_x = grid_x.contiguous().view(1, self.grid_size, self.grid_size, 1)
        self.grid_y = grid_y.contiguous().view(1, self.grid_size, self.grid_size, 1)

    def __call__(self, predictions, labels):
        targets = self._build_targets(labels).to(predictions.device)

        # Get predobj, predxy, predwh, predcls
        pred_obj = predictions[..., 0]
        pred_xy = predictions[..., 1:3]
        pred_wh = predictions[..., 3:5]
        pred_cls = predictions[..., 5:]
        # Get the same for target (obj + noobj)

        target_obj = (targets[..., 0] == 1).float()
        target_noobj = (targets[..., 0] == 0).float()
        target_xy = targets[..., 1:3]
        target_wh = targets[..., 3:5]
        target_cls = targets[..., 5]

        with torch.no_grad():
            iou_pred_with_target = self._calculate_iou(pred_box_cxcywh=predictions[..., 1:5], target_box_cxcywh=targets[..., 1:5])

        # calculate loss for oobj, noobj, xy, wh, cls
        obj_loss = self.obj_loss(pred_obj, iou_pred_with_target) * target_obj
        obj_loss = obj_loss.sum() / self.batch_size

        noobj_loss = self.obj_loss(pred_obj, pred_obj * 0) * target_noobj
        noobj_loss = noobj_loss.sum() / self.batch_size

        txty_loss = self.box_loss(pred_xy, target_xy).sum(dim=-1) * target_obj
        txty_loss = txty_loss.sum() / self.batch_size

        twth_loss = self.box_loss(pred_wh, target_wh).sum(dim=-1) * target_obj
        twth_loss = twth_loss.sum() / self.batch_size
        
        cls_loss = self.class_loss(pred_cls, target_cls) * target_obj
        cls_loss = cls_loss.sum() / self.batch_size

        multipart_loss = self.lambda_obj * obj_loss + noobj_loss + (txty_loss + twth_loss) + cls_loss
        return [multipart_loss, obj_loss, noobj_loss, txty_loss, twth_loss, cls_loss]



        
    def _calculate_iou(self, pred_box_cxcywh, target_box_cxcywh):
        pred_x1y1x2y2 = self.xywh_to_x1y1x2y2(pred_box_cxcywh)
        target_x1y1x2y2 = self.xywh_to_x1y1x2y2(target_box_cxcywh)

        inter_x1 = torch.max(pred_x1y1x2y2[..., 0], target_x1y1x2y2[..., 0])
        inter_y1 = torch.max(pred_x1y1x2y2[..., 1], target_x1y1x2y2[..., 1])
        inter_x2 = torch.min(pred_x1y1x2y2[..., 2], target_x1y1x2y2[..., 2])
        inter_y2 = torch.min(pred_x1y1x2y2[..., 3], target_x1y1x2y2[..., 3])
        
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        pred_area = (pred_x1y1x2y2[..., 2] - pred_x1y1x2y2[..., 0]) * (pred_x1y1x2y2[..., 3] - pred_x1y1x2y2[..., 1])
        target_area = (target_x1y1x2y2[..., 2] - target_x1y1x2y2[..., 0]) * (target_x1y1x2y2[..., 3] - target_x1y1x2y2[..., 1])
        union = abs(pred_area) + abs(target_area) - inter
        inter[inter.gt(0)] = inter[inter.gt(0)] / union[inter.gt(0)] # calculate iou only where intersection is greater than 0

        return inter



    def xywh_to_x1y1x2y2(self, boxes):
        # x y and are rel to grid -> recalculate to rel to whole
        x_rel_to_img = (boxes[..., 0] + self.grid_x.to(self.device)) / self.grid_size
        y_rel_to_img = (boxes[..., 1] + self.grid_y.to(self.device)) / self.grid_size
        # w h are refinements of the anchors -> recalculte the whole w h
        w = torch.exp(boxes[..., 2]) * self.anchors[:, 0].to(self.device) # TODO why this anchor??
        h = torch.exp(boxes[..., 3]) * self.anchors[:, 1].to(self.device) 
        # then just get xmin ymin xmax ymax from the above
        x1 = x_rel_to_img - w/2
        y1 = y_rel_to_img - h/2
        x2 = x_rel_to_img + w/2
        y2 = y_rel_to_img + h/2
        return torch.stack((x1, y1, x2, y2), dim=-1)

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





