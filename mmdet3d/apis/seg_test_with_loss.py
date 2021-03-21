import mmcv
import torch
from tools.evaluator import SegEvaluator


def seg_test_with_loss(model, data_loader):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    evaluator = SegEvaluator(class_names=dataset.SEG_CLASSES)
    print('\nStart Test Loop')
    batch_size = data_loader.batch_size
    seg_losses = []
    for idx, data in enumerate(data_loader):
        with torch.no_grad():
            seg_logits, seg_loss = model(return_loss=False, with_loss=True, **data)
            # return_loss=False: forward_test; with_loss=True: return seg_loss

        seg_losses.append(seg_loss)

        # handle seg
        seg_label = data['seg_label'].data[0]  # list of tensor
        seg_pts_indices = data['seg_pts_indices'].data[0]  # list of tensor
        seg_pred = seg_logits.argmax(1).cpu().numpy()
        pred_list = []
        gt_list = []
        left_idx = 0
        for i in range(len(seg_label)):
            # num_points = len(seg_pts_indices[i])
            assert len(seg_label[i]) == len(seg_pts_indices[i])
            num_points = len(seg_label[i])
            right_idx = left_idx + num_points
            pred_list.append(seg_pred[left_idx: right_idx])
            gt_list.append(seg_label[i].numpy())
            left_idx = right_idx
        evaluator.batch_update(pred_list, gt_list)

        # batch_size = len(box_res)
        for _ in range(batch_size):
            prog_bar.update()

    print(evaluator.print_table())
    print('overall_acc:', evaluator.overall_acc)
    print('overall_iou:', evaluator.overall_iou)
    print('seg_loss:', sum(seg_losses) / len(seg_losses))
    print()

