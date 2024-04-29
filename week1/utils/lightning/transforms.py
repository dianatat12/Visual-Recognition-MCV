import albumentations as A
import albumentations.pytorch.transforms as PT
import torchvision.transforms as transforms


def albumentations_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.3),
            A.GaussNoise(p=0.3),
            A.OneOf(
                [
                    A.MotionBlur(p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                ],
                p=0.3,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.ToGray(p=0.4),
            A.ImageCompression(p=0.3),
            A.ToSepia(p=0.2),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
        ]
    )


def preprocess(dims):
    return A.Compose(
        [
            A.Resize(dims[0], dims[1]),
            A.Normalize(),
            PT.ToTensorV2(),
        ]
    )


def contrast_transforms(image):
    augs = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=64),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    image = augs(image)

    return image
