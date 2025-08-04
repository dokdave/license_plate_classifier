from torchvision import transforms

def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
            transforms.GaussianBlur(kernel_size=(3, 3)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
