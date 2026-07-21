import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from Read_dataset import *
from Read_dataset.FourDME import FourDME_Dataset
from paired_transform import PairedFaceTransform
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import random
import numpy as np


def get_sampler(dataset, generator=None):
    labels = [int(np.load(os.path.join(s["folder"], "label.npy"))) for s in dataset.samples]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights),
                                  replacement=True, generator=generator)


def get_sampler_for_indices(dataset, indices, generator=None):
    """Same idea as get_sampler(), but restricted to a subset of indices --
    used for the per-fold train split in LOSO, where class balance should
    be computed only over the samples actually in that fold's train set."""
    labels = [int(np.load(dataset.samples[i]["label_path"])) for i in indices]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights),
                                  replacement=True, generator=generator)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(args):
    """Single train/val split datasets (unrelated to the LOSO refactor).
    4DME_MOTION is NOT handled here -- see get_loso_dataloaders() below,
    since LOSO returns a LIST of (train_loader, val_loader, held_out_subject)
    tuples instead of a single pair."""
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
    elif args.data_type == "4DME_MOTION":
        raise ValueError(
            "4DME_MOTION uses LOSO (subject-independent) evaluation -- call "
            "get_loso_dataloaders(args) instead of get_dataloaders(args), "
            "since it returns a list of per-fold (train_loader, val_loader, "
            "held_out_subject) tuples rather than a single pair."
        )
    else:
        cheo_root = os.path.join(data_dir, "cheo_dataset")
        train_dataset = DatasetCheo(json_path=os.path.join(cheo_root, "votes_train.json"), root_dir=cheo_root, transform=train_transform)
        val_dataset = DatasetCheo(json_path=os.path.join(cheo_root, "votes_valid.json"), root_dir=cheo_root, transform=val_transform)

    g = torch.Generator()
    g.manual_seed(args.seed)

    if getattr(args, "use_sampler", False):
        train_sampler = get_sampler(train_dataset, generator=g) 
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker, generator=g,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker, generator=g,
        )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader


def get_loso_dataloaders(args):
    """
    Builds true leave-one-subject-out splits for the 4DME_MOTION dataset:
    one fold per unique subject, that subject's samples held out as the
    validation set, everyone else's samples used for training.

    Returns a list of (train_loader, val_loader, held_out_subject) tuples.
    train.py is expected to loop over this list, train ONE model per fold
    from scratch, and average the resulting metrics across folds -- this
    is the standard LOSO protocol used across the ME literature (STAG,
    CausalNet, HTNet, etc. all report averaged metrics over subject folds,
    not one shared train/val split).

    Two FourDME_Dataset instances are built over the SAME root directory,
    one with the train transform (augmented) and one with the val
    transform (no augmentation) -- Subset() then indexes into whichever one
    is appropriate for train vs. val. Both instances list samples via the
    same sorted(os.listdir(...)) order, so indices line up 1:1 between them.
    """
    from sklearn.model_selection import LeaveOneGroupOut

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "Datasets")
    root = os.path.join(data_dir, "4dme_ma3d_motion")  # single merged folder,
                                                          # produced by the
                                                          # updated run_inference_flow.py

    train_tf = PairedFaceTransform(img_size=224, train=True)
    val_tf   = PairedFaceTransform(img_size=224, train=False)

    dataset_train_view = FourDME_Dataset(root, transform=train_tf, flow_key="flow_map", verbose=True)
    dataset_val_view   = FourDME_Dataset(root, transform=val_tf,   flow_key="flow_map")

    subjects = np.array(dataset_train_view.subjects)
    n_subjects = len(set(subjects.tolist()))
    print(f"[LOSO] {len(dataset_train_view)} samples across {n_subjects} subjects "
          f"-> {n_subjects} folds")

    logo = LeaveOneGroupOut()
    dummy_X = np.zeros(len(subjects))

    g = torch.Generator()
    g.manual_seed(args.seed)

    splits = []
    for train_idx, test_idx in logo.split(dummy_X, groups=subjects):
        held_out = str(subjects[test_idx][0])

        train_subset = Subset(dataset_train_view, train_idx)
        val_subset   = Subset(dataset_val_view, test_idx)

        if getattr(args, "use_sampler", False):
            sampler = get_sampler_for_indices(dataset_train_view, train_idx, generator=g)
            train_loader = DataLoader(
                train_subset, batch_size=args.batch_size, sampler=sampler,
                num_workers=args.num_workers, pin_memory=True, drop_last=True,
                worker_init_fn=seed_worker, generator=g,
            )
        else:
            train_loader = DataLoader(
                train_subset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True,
                worker_init_fn=seed_worker, generator=g,
            )

        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

        splits.append((train_loader, val_loader, held_out))

    debug_subject = getattr(args, "loso_debug_subject", None)
    if debug_subject:
        splits = [s for s in splits if s[2] == str(debug_subject)]
        if not splits:
            raise ValueError(
                f"--loso_debug_subject '{debug_subject}' not found among "
                f"subjects: {sorted(set(subjects.tolist()))}"
            )
        print(f"[LOSO] --loso_debug_subject set -> restricting to 1 fold "
              f"(held-out subject: {debug_subject})")

    return splits