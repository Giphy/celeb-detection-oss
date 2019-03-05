import logging
import os
import torch

import model_training.utils as utils
import torchvision.models as models

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dotenv import load_dotenv

# the following code must be invoked before model_training imports
# because of env variables in related modules
load_dotenv('.env')  # noqa: E402

from model_training.models.cnn.resnet import AdaptResNetBottleneck, ResNetCenterLoss
from model_training.preprocessors.datasets_builder import DatasetsBuilder
from model_training.dataset_loaders.facial_dataset import FacialDataset
from model_training.dataset_loaders.img_augmentor import Augmenter
from model_training.trainers.trainer import Trainer
from model_training.helpers.tensorboard_client import TensorboardClient


if __name__ == '__main__':
    use_cuda = os.getenv('USE_GPU') == 'true'  # noqa: E402

    experiment_dir = os.path.join(os.getenv('WORKDIR'), __file__.split('.')[0].split('/')[-1])
    utils.ensure_dir(experiment_dir)

    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join(experiment_dir, 'logs.log')),
                            logging.StreamHandler()
                        ])

    labels_file = os.path.join(experiment_dir, 'labels.csv')
    raw_dataset_dir = os.path.join(experiment_dir, 'raw_dataset')
    utils.ensure_dir(raw_dataset_dir)
    train_dataset_dir = os.path.join(experiment_dir, 'train_dataset')
    utils.ensure_dir(train_dataset_dir)
    val_dataset_dir = os.path.join(experiment_dir, 'val_dataset')
    utils.ensure_dir(val_dataset_dir)
    model_weights_dir = os.path.join(experiment_dir, 'model')
    utils.ensure_dir(model_weights_dir)

    datasets_builder = DatasetsBuilder(
        datasets=[raw_dataset_dir],
        train_dataset_path=train_dataset_dir,
        val_dataset_path=val_dataset_dir,
        val_split=0.1,
        detection_margin=0.1,
        use_cuda=use_cuda
    )
    datasets_builder.perform()
    del datasets_builder

    train_dataset = FacialDataset(
        dataset_path=train_dataset_dir,
        labels_file_path=labels_file,
        transform=Augmenter(augmentation_rate=0.3)
    )

    train_dataset_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    val_dataset = FacialDataset(
        dataset_path=val_dataset_dir,
        labels_file_path=labels_file,
        transform=Augmenter(augmentation_rate=0.3)
    )

    val_dataset_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)

    log_dir = os.path.join(experiment_dir, 'logs')
    summary_writer = SummaryWriter(log_dir=log_dir)
    tb_client = TensorboardClient(log_dir, port=int(os.getenv('TENSORBOARD_PORT', 5055)))
    tb_client.run()

    embedding_size = 256
    num_classes = len(utils.labels_by_name(labels_file))

    model = models.resnet50(num_classes=len(utils.labels_by_name(labels_file)))
    adapted_model = AdaptResNetBottleneck(model, embedding_size, num_classes)
    model_with_loss = ResNetCenterLoss(
        adapted_model,
        num_classes,
        embedding_size,
        center_loss_weight=0.001,
        use_cuda=use_cuda,
        summary_writer=summary_writer
    )

    optimizer = torch.optim.Adam(model_with_loss.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    trainer = Trainer(
        model=model_with_loss,
        data_loaders={'train': train_dataset_loader, 'val': val_dataset_loader},
        epochs=100,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        weights_path=model_weights_dir,
        use_cuda=use_cuda,
        log_per_batches=100,
        save_per_batches=2000
    )
    trainer.perform()
