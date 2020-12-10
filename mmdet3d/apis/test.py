import mmcv
import torch
from tools.evaluator import SegEvaluator


def mmda_single_gpu_test(model, data_loader, show=False, out_dir=None):
    model.eval()
    box_preds = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    evaluator = SegEvaluator(class_names=dataset.SEG_CLASSES)
    print('\nStart Test Loop')
    # print('batch_size:', data_loader.batch_size)  # 1
    for idx, data in enumerate(data_loader):
        # print(type(data['img'][0]))  # DataContainter
        with torch.no_grad():
            seg_res, box_res = model(return_loss=False, rescale=True, **data)
        # len(box_res) == batch_size
        # print(box_res[0].keys())  # 'pts_bbox'

        # handle seg
        seg_label = data['seg_label'][0].data[0]  # list of tensor
        seg_pts_indices = data['seg_pts_indices'][0].data[0]  # list of tensor
        seg_pred = seg_res.argmax(1).cpu().numpy()
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

        # handle box
        if show:
            model.module.show_results(data, box_res, out_dir)

        box_preds.extend(box_res)

        # progress bar
        batch_size = len(box_res)
        for _ in range(batch_size):
            prog_bar.update()

    print(evaluator.print_table())
    print('overall_acc:', evaluator.overall_acc)
    print('overall_iou:', evaluator.overall_iou)
    return box_preds  # list of dict; key='pts_bbox'


def single_gpu_test(model, data_loader, show=False, out_dir=None):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            model.module.show_results(data, result, out_dir)

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
