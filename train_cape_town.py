import os
import albumentations as A

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp

from segmentation import Dataset
from segmentation import SolarModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger




# Clear GPU cache
# torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,5,6,7'


# Paths
CP_DATA_DIR = "/home/as1233/data/cape_town"

cp_x_train_dir = os.path.join(CP_DATA_DIR, 'train/images')
cp_y_train_dir = os.path.join(CP_DATA_DIR, 'train/masks')

cp_x_valid_dir = os.path.join(CP_DATA_DIR, 'val/images')
cp_y_valid_dir = os.path.join(CP_DATA_DIR, 'val/masks')

cp_x_test_dir = os.path.join(CP_DATA_DIR, 'test/images')
cp_y_test_dir = os.path.join(CP_DATA_DIR, 'test/masks')

cp_tile_size = 896





# Data augmentation
def get_training_augmentation(tile_size):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.2, rotate_limit=15, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=tile_size, min_width=tile_size, always_apply=True),
        A.RandomCrop(height=tile_size, width=tile_size, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)

# Validation set images augmentation
def get_validation_augmentation(tile_size):
    """Ensure validation images are correctly sized."""
    test_transform = [
        A.PadIfNeeded(min_height=tile_size, min_width=tile_size, always_apply=True),
    ]
    return A.Compose(test_transform)





# Load datasets
CLASSES = ["solar_panel"]


cp_train_dataset = Dataset(
    cp_x_train_dir,
    cp_y_train_dir,
    augmentation=get_training_augmentation(cp_tile_size),
    classes=CLASSES,
)

cp_valid_dataset = Dataset(
    cp_x_valid_dir,
    cp_y_valid_dir,
    augmentation=get_validation_augmentation(cp_tile_size),
    classes=CLASSES,
)

cp_test_dataset = Dataset(
    cp_x_test_dir,
    cp_y_test_dir,
    augmentation=get_validation_augmentation(cp_tile_size),
    classes=CLASSES,
)

cp_train_loader = DataLoader(cp_train_dataset, batch_size=16, shuffle=True, num_workers=4)
cp_valid_loader = DataLoader(cp_valid_dataset, batch_size=16, shuffle=False, num_workers=4)
cp_test_loader = DataLoader(cp_test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Define constants
EPOCHS = 20
T_MAX = EPOCHS * len(cp_train_loader)
OUT_CLASSES = 1

csv_logger = CSVLogger(save_dir='/home/as1233/SolarDetection/logs', name='solar_model_training')

# Load the trained model on california dataset
model = SolarModel.load_from_checkpoint(
    checkpoint_path="/home/as1233/SolarDetection/trained_models/solar_model_checkpoint_124/epoch=0-val_loss=0.00.ckpt",
    arch='FPN',
    encoder_name='resnext50_32x4d',
    in_channels=3,
    out_classes=OUT_CLASSES,
    T_max = T_MAX
)

# Move the model to GPU
model.to('cuda')


checkpoint_callback = ModelCheckpoint(
    dirpath='/home/as1233/SolarDetection/trained_models/solar_model_checkpoint_124_fine_tuned',   
    filename='{epoch}-{val_loss:.2f}',  
    monitor='valid_dataset_iou',                
    save_top_k=1,                       
    mode='min',                         
)


trainer = pl.Trainer(
    max_epochs=EPOCHS,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
    devices=-1,  
    accelerator='gpu',  
    strategy='ddp',  
    precision=16,  
    accumulate_grad_batches=4,  
    logger=csv_logger  
)

# Train the model
trainer.fit(
    model,
    train_dataloaders=cp_train_loader,
    val_dataloaders=cp_valid_loader,
)


# print eval metrics 
model.eval()

valid_metrics = trainer.validate(model, dataloaders=cp_valid_loader, verbose=False)
print(valid_metrics)


# print test metrics
test_metrics = trainer.test(model, dataloaders=cp_test_loader, verbose=False)
print(test_metrics)
