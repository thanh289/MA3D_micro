"""Evaluate a saved motion model using its recorded architecture configuration."""
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from models.MA3D import MA3D
from build_dataloader import _LOSO_DATASET_REGISTRY
from paired_transform import PairedFaceTransform
from engine import validate
from sklearn.metrics import f1_score, recall_score


def load_model(ckpt_path, device, num_classes=None):
    # Checkpoints produced by the training scripts contain model tensors and metadata.
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    config = dict(state.get('model_config') or {})
    if not config:
        raise ValueError('Checkpoint lacks model_config; architecture cannot be inferred safely')
    if num_classes is not None and config['num_classes'] != num_classes:
        raise ValueError('Requested class count differs from checkpoint')
    # Pre-change checkpoints have no explicit aggregation selector.
    config.setdefault('motion_aggregation', 'legacy_pyramid')
    config['landmark_checkpoint'] = None  # Backbone weights are inside the full checkpoint.
    model = MA3D(**config)
    # Legacy geometry=none still saved an unused frozen backbone in the original ZIP.
    weights = state['model']
    if model.face_landback is None and config['motion_aggregation'] == 'legacy_pyramid':
        weights = {k: v for k, v in weights.items() if not k.startswith('face_landback.')}
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    return validate(model, loader, torch.nn.CrossEntropyLoss(), device, 0, 1)


def main():
    parser = argparse.ArgumentParser('Evaluate a motion checkpoint')
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--data_type', choices=list(_LOSO_DATASET_REGISTRY), default='4DME_MOTION')
    parser.add_argument('--data_root', required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--subject', help='Held-out subject of this LOSO checkpoint')
    selection.add_argument('--all_samples', action='store_true',
                           help='Evaluate a complete external target; do not use on LOSO training subjects')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.ckpt_path, device)
    config = model.model_config
    dataset_cls, _ = _LOSO_DATASET_REGISTRY[args.data_type]
    # Standalone rise-only inference does not require offset/fall files.
    needs_fall = config['use_rise_fall'] and config['rise_fall_mode'] == 'feature_gate'
    dataset = dataset_cls(args.data_root, transform=PairedFaceTransform(train=False),
                          use_rise_fall=needs_fall, load_fall=needs_fall,
                          load_au=config['use_au'], file_suffix='')
    if args.subject is not None:
        indices = [i for i, subject in enumerate(dataset.subjects) if str(subject) == args.subject]
        dataset = Subset(dataset, indices)
    if not len(dataset):
        raise ValueError('No evaluation samples match this dataset/subject')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    loss, acc, labels, preds = evaluate(model, loader, device)
    classes = list(range(config['num_classes']))
    uf1 = f1_score(labels, preds, labels=classes, average='macro', zero_division=0)
    uar = recall_score(labels, preds, labels=classes, average='macro', zero_division=0)
    print(f'Loss={loss:.6f} Acc={acc:.4f} UF1={uf1:.4f} UAR={uar:.4f} N={len(labels)}')


if __name__ == '__main__':
    main()
