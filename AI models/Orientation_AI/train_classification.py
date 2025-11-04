import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.functional as tf
import torch.utils.data
import time
from tqdm import tqdm
import model
import argparse
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
import sys
import os
import pathlib
import glob
import cv2
from PIL import Image
import albumentations as A

# Global variable for image folder (used by Training class)
image_folder_global = None


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set. It creates 360 bins for the angles.    
    """
    def __init__(self, images, angles, n_training):
        """
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()

        self.n_training = n_training

        self.angles = angles
        self.n_images = self.angles.shape[0]
        self.images = images
        
        self.stheta = self.angles[:, 1]
        self.ctheta = self.angles[:, 2]

        self.mn = np.zeros((self.n_images))
        self.std = np.zeros((self.n_images))
        for i in range(self.n_images):
            self.mn[i] = np.mean(self.images[i], axis=(0,1))
            self.std[i] = np.std(self.images[i], axis=(0,1))

        self.transform = A.Compose([A.RandomToneCurve(scale=0.2, p=0.5), A.RandomBrightnessContrast(p=0.2),])

        # use 360 bins 1 g
        self.bins = np.arange(360) * 1.0 - 180.0
        
        # for i in range(self.n_images):
            # self.images[i] = (self.images[i] - self.mn[None, None, :]) / self.std[None, None, :]
                
    def __getitem__(self, index):

        t = np.random.choice(['none', 'flipx', 'flipy', 'rot90', 'rot180', 'rot270'])

        ind = np.random.randint(low=0, high=self.n_images)
        
        image = self.images[ind]
        ctheta = self.ctheta[ind]
        stheta = self.stheta[ind]

        # None
        if (t == 'none'):
            image_out = image
            ctheta_out = ctheta
            stheta_out = stheta

        # Flip X
        if (t == 'flipx'):
            image_out = np.fliplr(image)
            ctheta_out = -ctheta
            stheta_out = stheta

        # Flip Y
        if (t == 'flipy'):
            image_out = np.flipud(image)
            ctheta_out = ctheta
            stheta_out = -stheta

        # Rotation 90 deg
        if (t == 'rot90'):
            image_out = np.rot90(image, k=1)
            ctheta_out = -stheta
            stheta_out = ctheta

        # Rotation 180 deg
        if (t == 'rot180'):
            image_out = np.rot90(image, k=2)
            ctheta_out = -ctheta
            stheta_out = -stheta

        # Rotation 270 deg
        if (t == 'rot270'):
            image_out = np.rot90(image, k=3)
            ctheta_out = stheta
            stheta_out = -ctheta

        image_out = self.transform(image=image_out)['image']

        image_out = (image_out - self.mn[ind]) / self.std[ind]

        angle = np.arctan2(stheta_out, ctheta_out) * 180.0 / np.pi

        ind = np.digitize(angle, self.bins) - 1
        
        return np.expand_dims(image_out, 0).astype('float32'), ind.astype('long'), angle

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')
        
# a simple custom collate function, just to show the idea
def my_collate(batch):
    image = [torch.tensor(item[0]) for item in batch]
    cstheta = [torch.tensor(item[1]) for item in batch]
    angle = [torch.tensor(item[2]) for item in batch]
    return [image, cstheta, angle]

class Training(object):
    """
    Training class used for building the model classy the orientation of new images.    
    """
    def __init__(self, angle_file, batch_size, validation_split=0.2, gpu=0, smooth=0.05, training_size=10000, validation_size=3000, checkpoint=None, image_folder=None):
        global image_folder_global
        
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.smooth = smooth
        self.device = torch.device(f"cuda" if self.cuda else "cpu") #:cuda:{self.gpu}
        training_split = 0.8

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size
        self.validation_split = validation_split        
        
        kwargs = {'num_workers': 1, 'pin_memory': False} if self.cuda else {}        
        
        self.model = model.Network(in_planes=1, n_kernel=16, n_out=360).to(self.device)
        
        if (checkpoint != None):
            self.checkpoint = '{0}'.format(checkpoint)
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("loaded checkpoint '{}'".format(self.checkpoint))
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # Set global image_folder if provided
        if image_folder is not None:
            image_folder_global = image_folder
        
        if angle_file is None:
            raise Exception("Required angle file missing.")
 
        angles = np.loadtxt(angle_file)
        n_images = angles.shape[0]
        images_orig = [None] * n_images

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        print("Reading images...")
        # Use global image_folder if available, otherwise try to find it
        if image_folder_global is None:
            # Try to find image folder from angle_file location
            image_folder_global = os.path.dirname(angle_file) if angle_file else '.'
        
        for i in tqdm(range(n_images)):
            image_path = os.path.join(image_folder_global, f'{i:04d}.png')
            if not os.path.exists(image_path):
                # Try jpg format
                image_path = os.path.join(image_folder_global, f'{i:04d}.jpg')
            if not os.path.exists(image_path):
                # Try to find any image with index in name
                import glob
                pattern = os.path.join(image_folder_global, f'*{i:04d}.*')
                matches = glob.glob(pattern)
                if matches:
                    image_path = matches[0]
                else:
                    raise FileNotFoundError(f"Image not found for index {i}: {image_path}")
            
            images_orig[i] = np.array(Image.open(image_path).convert("L"))
            # tmp = cv2.cvtColor(images[i], cv2.COLOR_RGB2YCrCb)
            # tmp[:, :, 0] = clahe.apply(tmp[:, :, 0])
            # images[i] = cv2.cvtColor(tmp, cv2.COLOR_YCrCb2RGB)
            # images[i] = images[i] / 255.0

        # Shuffle the order
        indices = np.arange(angles.shape[0])
        np.random.shuffle(indices)

        angles = angles[indices, :]
        images = [images_orig[i] for i in indices]
        
        self.training_size = training_size
        self.validation_size = validation_size
        
        self.train_dataset = Dataset(images[0:int(training_split*n_images)], angles[0:int(training_split*n_images), :], n_training=training_size)
        self.validation_dataset = Dataset(images[int(training_split*n_images):], angles[int(training_split*n_images):, :], n_training=validation_size)
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=my_collate, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=my_collate, **kwargs)
        
    def init_optimize(self, epochs, lr, weight_decay, scheduler, output_dir=None):

        self.lr = lr
        self.weight_decay = weight_decay        
        print('Learning rate : {0}'.format(lr))
        self.n_epochs = epochs
        
        if output_dir is None:
            p = pathlib.Path('trained/')
            p.mkdir(parents=True, exist_ok=True)
            current_time = time.strftime("%Y-%m-%d-%H_%M_%S")
            self.out_name = 'trained/{0}'.format(current_time)
        else:
            # Use provided output directory
            p = pathlib.Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            current_time = time.strftime("%Y-%m-%d-%H_%M_%S")
            self.out_name = os.path.join(output_dir, '{0}'.format(current_time))

        # Copy model
        file = model.__file__.split('\\')[-1]
        shutil.copyfile(model.__file__, '{0}_model.py'.format(self.out_name))
        shutil.copyfile('{0}/{1}'.format(os.path.dirname(os.path.abspath(__file__)), file), '{0}_trainer.py'.format(self.out_name))
        self.file_mode = 'w'

        f = open('{0}_hyper.dat'.format(self.out_name), 'w')
        f.write('Learning_rate       Weight_decay     \n')
        f.write('{0}    {1}'.format(self.lr, self.weight_decay))
        f.close()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler, gamma=0.5)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, eta_min=self.lr / 50.0)

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = -1e10

        trainF = open('{0}.loss.csv'.format(self.out_name), self.file_mode)

        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):            
            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()

            trainF.write('{},{},{}\n'.format(
                epoch, self.loss[-1], self.loss_val[-1]))
            trainF.flush()

            is_best = self.loss_val[-1] < best_loss
            best_loss = min(self.loss_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, filename='{0}.pth'.format(self.out_name))

        trainF.close()

    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        correct = torch.zeros(self.training_size)
        predicted = torch.zeros(self.training_size)
        
        loop = 0
        
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (images, cstheta, angle) in enumerate(t):
            
            self.optimizer.zero_grad()

            loss = 0.0            

            for i in range(len(images)):
                image = images[i][None, ...].to(self.device)
                cst = cstheta[i].to(self.device)

                out = self.model(image)       
                # Loss
                cst=cst.type(torch.int64)
                loss += self.loss_fn(out.cuda(), cst.unsqueeze(0))

                correct[loop] = cst
                predicted[loop] = out.max(1)[1]

                loop += 1

            pct = 100.0*predicted[0:loop].eq(correct[0:loop].view_as(predicted[0:loop])).sum().item() / loop

            loss.backward()

            self.optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            if (NVIDIA_SMI):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=tmp.gpu, mem=tmp.memory, pct=pct)
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr, pct=pct)
            
        self.loss.append(loss_avg)
        
    def test(self, epoch):
        self.model.eval()
        t = tqdm(self.validation_loader)        
        loss_avg = 0.0
        loop = 0

        correct = torch.zeros(self.validation_size)
        predicted = torch.zeros(self.validation_size)

        with torch.no_grad():
            for batch_idx, (images, cstheta, angle) in enumerate(t):
            
                loss = 0.0
                for i in range(len(images)):
                    image = images[i][None, ...].to(self.device)
                    cst = cstheta[i].to(self.device)

                    out = self.model(image)
                    
                    # Loss
                    cst=cst.type(torch.int64)
                    loss += self.loss_fn(out, cst.unsqueeze(0))

                    correct[loop] = cst
                    predicted[loop] = out.max(1)[1]

                    loop += 1

                pct = 100.0*predicted[0:loop].eq(correct[0:loop].view_as(predicted[0:loop])).sum().item() / loop

                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
            
                t.set_postfix(loss=loss_avg, pct=pct)
            
        self.loss_val.append(loss_avg)

def train_orientation_model(dataset_path, epochs, batch_size, output_dir, angle_file=None, image_folder=None):
    """
    Wrapper function for training orientation model from Flask.
    
    Args:
        dataset_path: Path to the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Directory to save results
        angle_file: Optional path to angle file. If None, will try to find/create one.
        image_folder: Optional path to image folder. If None, will use dataset_path.
    """
    import glob
    
    # Find angle file if not provided
    if angle_file is None:
        # Look for angle files in the dataset
        angle_files = glob.glob(os.path.join(dataset_path, '**', '*.txt'), recursive=True)
        angle_files += glob.glob(os.path.join(dataset_path, '**', 'angles*.txt'), recursive=True)
        if angle_files:
            angle_file = angle_files[0]
        else:
            # Create a dummy angle file if none found
            # This assumes images are named 0000.png, 0001.png, etc.
            image_files = sorted(glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True))
            image_files += sorted(glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True))
            
            if not image_files:
                raise ValueError(f"No images found in dataset path: {dataset_path}")
            
            # Create dummy angle file (id, x_diff, y_diff, angle)
            angle_file = os.path.join(output_dir, 'angles.txt')
            with open(angle_file, 'w') as f:
                for i in range(len(image_files)):
                    # Dummy values - in real use, these should come from your data
                    f.write(f"{i} 0.0 0.0 0.0\n")
            print(f"Created dummy angle file at {angle_file}. Please provide a real angle file for accurate training.")
    
    # Find image folder if not provided
    if image_folder is None:
        # Look for images directory
        image_dirs = []
        try:
            for root, dirs, files in os.walk(dataset_path):
                if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                    image_dirs.append(root)
                    break  # Use first directory found with images
        except (OSError, PermissionError) as e:
            print(f"Warning: Error walking dataset path: {e}")
        
        if image_dirs:
            image_folder = image_dirs[0]  # Use first directory with images
        else:
            image_folder = dataset_path
            print(f"Warning: No image directory found, using dataset_path: {dataset_path}")
    
    # Set global image_folder for the Training class
    global image_folder_global
    image_folder_global = image_folder
    
    # Change to output directory for saving models
    original_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize training
        deepnet = Training(
            angle_file=angle_file,
            batch_size=batch_size,
            gpu=0,  # Use first GPU or CPU
            smooth=0.05,
            checkpoint=None,
            image_folder=image_folder
        )
        
        # Initialize optimizer with output directory
        deepnet.init_optimize(
            epochs=epochs,
            lr=3e-4,  # Default learning rate
            weight_decay=1e-4,  # Default weight decay
            scheduler=10,  # Default scheduler
            output_dir=output_dir
        )
        
        # Train
        print("Starting orientation model training...")
        deepnet.optimize()
        print("Orientation model training completed successfully.")
        
    finally:
        os.chdir(original_dir)


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--af', '--angle-file',
                    metavar='AF', help='Angle file. Format: (id, x diff, y diff, angle in degrees)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--wd', '--weigth-decay', default=1e-4, type=float,
                    metavar='WD', help='Weigth decay')    
    parser.add_argument('--gpu', '--gpu', default=1, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=100, type=int,
                    metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--scheduler', '--scheduler', default=10, type=int,
                    metavar='SCHEDULER', help='Number of epochs before applying scheduler')
    parser.add_argument('--batch', '--batch', default=64, type=int,
                    metavar='BATCH', help='Batch size')
    parser.add_argument('--checkpoint', '--checkpoint-algorithm',
                    metavar='CHECKPOINT', help='Checkpoint algorithm from which the model starts training')
    parsed = vars(parser.parse_args())

    deepnet = Training(angle_file=parsed['af'], batch_size=parsed['batch'], gpu=parsed['gpu'], smooth=parsed['smooth'], checkpoint=parsed['checkpoint'])

    deepnet.init_optimize(parsed['epochs'], lr=parsed['lr'], weight_decay=parsed['wd'], scheduler=parsed['scheduler'], output_dir=None)
    deepnet.optimize()