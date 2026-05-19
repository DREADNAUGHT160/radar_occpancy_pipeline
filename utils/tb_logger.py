"""TensorBoard image and summary logging helpers."""
import torch


def log_train_images(writer, images, labels, outputs, epoch):
    if writer is None:
        return
    # Power — max-project over Doppler
    pow_bev = torch.max(images[:, 0, ...], dim=1, keepdim=True)[0]
    pow_bev = (pow_bev - pow_bev.min()) / (pow_bev.max() - pow_bev.min() + 1e-6)
    writer.add_images('Train/Input_Power', pow_bev, epoch)

    if images.shape[1] > 1:
        elev_bev = torch.max(images[:, 1, ...], dim=1, keepdim=True)[0]
        elev_bev = (elev_bev - elev_bev.min()) / (elev_bev.max() - elev_bev.min() + 1e-6)
        writer.add_images('Train/Input_Elevation', elev_bev, epoch)

    label_bev = torch.max(labels, dim=1, keepdim=True)[0] if labels.ndim == 4 else labels.unsqueeze(1)
    writer.add_images('Train/GT_Label', label_bev, epoch)

    with torch.no_grad():
        prob     = torch.sigmoid(outputs).detach()
        pred_bev = torch.max(prob, dim=1, keepdim=True)[0] if prob.ndim == 4 else prob.unsqueeze(1)
    writer.add_images('Train/Prediction', pred_bev, epoch)


def log_eval_images(writer, images, labels, preds_prob, mode, epoch):
    if writer is None:
        return
    pow_bev = torch.max(images[:, 0, ...], dim=1, keepdim=True)[0]
    pow_bev = (pow_bev - pow_bev.min()) / (pow_bev.max() - pow_bev.min() + 1e-6)
    writer.add_images(f'{mode}/Input_Power', pow_bev, epoch)

    if images.shape[1] > 1:
        elev_bev = torch.max(images[:, 1, ...], dim=1, keepdim=True)[0]
        elev_bev = (elev_bev - elev_bev.min()) / (elev_bev.max() - elev_bev.min() + 1e-6)
        writer.add_images(f'{mode}/Input_Elevation', elev_bev, epoch)

    label_bev = torch.max(labels, dim=1, keepdim=True)[0] if labels.ndim == 4 else labels.unsqueeze(1)
    pred_bev  = torch.max(preds_prob, dim=1, keepdim=True)[0] if preds_prob.ndim == 4 else preds_prob.unsqueeze(1)
    writer.add_images(f'{mode}/GT_Label',   label_bev, epoch)
    writer.add_images(f'{mode}/Prediction', pred_bev,  epoch)


def log_epoch_summary(writer, epoch, train_loss, val_loss, val_acc,
                      val_iou, val_prec, val_rec, lr, total_epochs):
    if writer is None:
        return
    summary = (
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Epoch | {epoch} / {total_epochs} |\n"
        f"| Train Loss | {train_loss:.6f} |\n"
        f"| Val Loss | {val_loss:.6f} |\n"
        f"| IoU | {val_iou:.4f} |\n"
        f"| Precision | {val_prec:.4f} |\n"
        f"| Recall | {val_rec:.4f} |\n"
        f"| Accuracy | {val_acc:.4f} |\n"
        f"| LR | {lr:.6f} |"
    )
    writer.add_text('Epoch_Summary', summary, epoch)
