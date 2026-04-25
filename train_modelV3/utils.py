import torchvision.transforms as T
import matplotlib.pyplot as plt

# --- Pad to square ---
def pad_to_square(img):
    w, h = img.size
    max_dim = max(w, h)

    pad_w = max_dim - w
    pad_h = max_dim - h

    padding = (
        pad_w // 2,
        pad_h // 2,
        pad_w - pad_w // 2,
        pad_h - pad_h // 2
    )

    return T.functional.pad(img, padding, fill=0)

# --- Train transforms ---
def get_train_transform():
    return T.Compose([
        T.Lambda(pad_to_square),
        T.Resize((224, 224)),

        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(5),

        T.RandomApply([
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.02
            )
        ], p=0.3),

        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# --- Val/test transforms ---
def get_val_transform():
    return T.Compose([
        T.Lambda(pad_to_square),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# --- Plot loss ---
def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")
    plt.close()