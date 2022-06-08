import argparse

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
from torch import optim, nn
import random, os
import numpy as np
from tqdm import tqdm

from datasets import SiameseDataset
from model.siamese_network import SiameseNetwork


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(model, dataloader, config, args):

    criterion = nn.TripletMarginLoss(margin=config["model_training"]["loss_margin"],
                                     p=config["model_training"]["loss_p"])
    optimizer = optim.Adam(model.parameters(), lr=config["model_training"]["learning_rate"])
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")

    iteration_number = 0
    loss_val = 0

    for epoch in range(args.epochs_number):

        # Iterate over batches
        for i, (img0, img1, img2) in tqdm(enumerate(dataloader), total=len(dataloader)):

            img0, img1, img2 = img0.to(device), img1.to(device), img2.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2, output3 = model(img0, img1, img2)

            # Pass the outputs of the networks and label into the loss function
            loss = criterion(output1, output2, output3)
            loss_val += loss.item()
            # Calculate the backpropagation
            loss.backward()

            # Optimize
            optimizer.step()

            # Every 10 batches print out the loss
            if i % 1000 == 0:
                print(f"Epoch number {epoch}\n Current loss {loss_val / 1000}\n")
                loss_val = 0
                iteration_number += 1000


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_dir', default='data/marble/train')
    parser.add_argument('--checkpoint_path_to_load', default='')
    parser.add_argument('--checkpoint_path_to_save', default='')
    parser.add_argument('--checkpoint_name', default='siamese')
    parser.add_argument('--epochs_number', default=5)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--config', default='config.yml')

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    seed_everything(42)

    device = torch.device(args.device)
    model = SiameseNetwork().to(device)

    if args.checkpoint_path_to_load != '':
        checkpoint = torch.load(args.checkpoint_path_to_load, map_location=torch.device(args.device))
        model.load_state_dict(checkpoint['model'])
    model.train()

    folder_dataset_train = datasets.ImageFolder(root=args.input_data_dir)
    transformation = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                         ])
    siamese_dataset = SiameseDataset(imageFolderDataset=folder_dataset_train,
                                     transform=transformation)
    train_dataloader = DataLoader(siamese_dataset, batch_size=args.batch_size, shuffle=True)

    train(model, train_dataloader, config, args)

    torch.save({
        'epoch': args.epochs_number,
        'model': model.state_dict(),
    }, f"{args.checkpoint_path_to_save}/{args.checkpoint_name}_{args.epochs_number}_epochs.pt")


if __name__ == '__main__':
    main()
