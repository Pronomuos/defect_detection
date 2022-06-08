import argparse

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import SiameseDataset
from model.siamese_network import SiameseNetwork
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random, os
import numpy as np


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_dataset(root_dir):
    folder_dataset = datasets.ImageFolder(root=root_dir)

    transformation = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                         ])

    siamese_dataset = SiameseDataset(imageFolderDataset=folder_dataset,
                                     transform=transformation)

    return siamese_dataset


def get_trainer(root_dir, batch_size=1):
    siamese_dataset = get_dataset(root_dir)
    dataloader = DataLoader(siamese_dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def get_similarity_predictions(dataloader, model):
    in_labels = []
    in_pred = []
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")

    for i, (img0, img1, img2) in tqdm(enumerate(dataloader), total=len(dataloader)):
        output1, output2, output3 = model(img0.to(device), img1.to(device), img2.to(device))
        sim_dist = F.pairwise_distance(output1, output2).item()
        dif_dist = F.pairwise_distance(output1, output3).item()
        in_pred.append(sim_dist)
        in_pred.append(dif_dist)
        in_labels.append(0)
        in_labels.append(1)
    return in_labels, in_pred


def get_best_threshold(preds, labels):
    min_val = min(preds)
    max_val = max(preds)

    range_val = np.linspace(min_val, max_val, 10)
    best = [0, 0]

    for val in range_val:
        temp_pred = list(map(lambda x: int(x > val), preds))
        score = accuracy_score(labels, temp_pred)

        if score > best[0]:
            best[0] = score
            best[1] = val
    return best


def evaluate_similarity(dataset_dir, model, batch_size):
    """
    1) The function calculates similarity between (l1 norm distance) three patches (tho of them are from
    the same class, the third one is from another class) in training set.
    2) Gets best threshold for the set.
    3) Calculates similarity in testing set, then uses the threshold.
    4) Prints a classification report (f1_score) on how often the model understands that
    two patches are from different classes or from the same one.
    """
    test_dataloader = get_trainer(os.path.join(dataset_dir, "test"), batch_size)
    train_dataloader = get_trainer(os.path.join(dataset_dir, "train"), batch_size)

    model.eval()

    labels, pred = get_similarity_predictions(train_dataloader, model)
    _, threshold = get_best_threshold(pred, labels)

    labels, pred = get_similarity_predictions(test_dataloader, model)
    pred = list(map(lambda x: int(x > threshold), pred))

    target_names = ['not_equal', 'equal']

    print(dataset_dir)
    print(f"Threshold - {threshold}.")
    print(classification_report(labels, pred, target_names=target_names))


def get_class_predictions(model, train_dataset, test_dataset):
    labels = []
    predictions = []
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")

    for i in tqdm(range(len(train_dataset)), total=len(train_dataset)):
        img = train_dataset.get_image(i)
        labels.append(img[1])
        img_embed = model.forward_once(img[0].to(device).unsqueeze(0))

        other_imgs = test_dataset.get_random_imgs()

        dist = []
        for other_img in other_imgs:
            other_img_embed = model.forward_once(other_img[0].to(device).unsqueeze(0))
            dist.append((F.pairwise_distance(img_embed, other_img_embed).item(), other_img[1]))
        dist.sort(key=lambda tup: tup[0], reverse=False)
        predictions.append(dist[0][1])

    return labels, predictions


def evaluate_class_predictions(dataset_dir, model):
    """
    For every patch in testing set we get patches of all classes in training set on random
    to evaluate similarity between them, then the patch becomes of class with the lowest value (l1 norm distance).
    Then we get a report about classification.
    """
    test_dataset = get_dataset(os.path.join(dataset_dir, "test"))
    train_dataset = get_dataset(os.path.join(dataset_dir, "train"))

    model.eval()

    labels, predictions = get_class_predictions(model, train_dataset, test_dataset)

    print(dataset_dir)
    print(classification_report(labels, predictions))


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_dir', default='data/marble')
    parser.add_argument('--checkpoint_path', default='model_checkpoints/siamese_1-4_mixed_36_epochs.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', default=1)

    args = parser.parse_args()

    seed_everything(42)
    device = torch.device(args.device)
    model = SiameseNetwork().to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    evaluate_similarity(args.input_data_dir, model, args.batch_size)
    evaluate_class_predictions(args.input_data_dir, model)


if __name__ == '__main__':
    main()
