import os
from torch.utils.data import DataLoader
from torchvision import transforms
from Read_dataset import *
from torch.utils.data import WeightedRandomSampler
from collections import Counter

def get_sampler(dataset):
    labels = [int(dataset[i]["label"]) for i in range(len(dataset))]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def get_dataloaders(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=1, scale=(0.05, 0.05))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "Datasets")

    if args.data_type == "RAF-DB":
        raf_path = os.path.join(data_dir, "RAF-DB_lmk")
        train_dataset = RAFDataset(raf_path, is_train=True, transform=train_transform)
        val_dataset = RAFDataset(raf_path, is_train=False, transform=val_transform)

    elif args.data_type == "VKIST":
        vkist_root = os.path.join(data_dir, "3087data_subject_split")
        stats_path = os.path.join(vkist_root, "micro_sb_emoca_stats.npz")
        train_dataset = VKIST_Dataset(vkist_root, is_train=True, transform=train_transform, stats_path=stats_path)
        val_dataset = VKIST_Dataset(vkist_root, is_train=False, transform=val_transform, stats_path=stats_path)
    elif args.data_type == "FerPlus":
        ferPlus_root = os.path.join(data_dir, "FER_Plus/fer_plus_05")
        train_dataset = FERPlusDataset(ferPlus_root, split="train", transform=train_transform)
        val_dataset = FERPlusDataset(ferPlus_root, split="test", transform=val_transform)
    elif args.data_type == "Caers":
        caers_root = os.path.join(data_dir, "caers/caers")
        train_dataset = CaersDataset(caers_root, split="train", transform=train_transform)
        val_dataset = CaersDataset(caers_root, split="test", transform=val_transform)
    elif args.data_type == "CheoFaMo":
        CheoFamo_root = os.path.join(data_dir, "CheoFamo")
        train_dataset = CheoFaMo(CheoFamo_root, split="train", transform=train_transform)
        val_dataset = CheoFaMo(CheoFamo_root, split="test", transform=val_transform)
    elif args.data_type == "4DME":
        dme_root   = os.path.join(data_dir, "4dme_ma3d")
        stats_path = os.path.join(dme_root, "4dme_stats.npz")
        train_dataset = FourDME_Dataset(dme_root, is_train=True,  transform=train_transform, stats_path=stats_path)
        val_dataset   = FourDME_Dataset(dme_root, is_train=False, transform=val_transform,   stats_path=stats_path)
    else:
        cheo_root = os.path.join(data_dir, "cheo_dataset")
        train_dataset = DatasetCheo(json_path=os.path.join(cheo_root, "votes_train.json"), root_dir=cheo_root, transform=train_transform)
        val_dataset = DatasetCheo(json_path=os.path.join(cheo_root, "votes_valid.json"), root_dir=cheo_root, transform=val_transform)

    sampler = get_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader