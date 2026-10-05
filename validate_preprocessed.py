"""Validate a complete 4DME preprocessing cohort before training."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def validate_dataset(root, require_au=True):
    root=Path(root)
    with (root/'dataset_info.json').open(encoding='utf-8') as f:
        info=json.load(f)
    count=info['num_classes']; R=info['n_roi']
    assert count in (3,5), 'Unsupported class protocol'
    assert len(info['class_names']) == count
    assert info['feature_mode'] == 'cnn', 'ROI CNN training requires spatial maps'
    folders=info['saved_folders']; assert folders and len(set(folders)) == len(folders), 'Empty/duplicate cohort'
    reselected_path=root/'dataset_info_gamdss.json'
    reselected=set()
    if reselected_path.exists():
        corrected=json.loads(reselected_path.read_text())
        for key in ['num_classes','class_names','roi_mode','n_roi','motion_mode','feature_mode','au_dim']:
            assert corrected[key] == info[key], f'Reselected configuration mismatch: {key}'
        reselected=set(corrected['saved_folders'])
        assert reselected <= set(folders), 'Reselected samples outside original cohort'
    distribution=np.zeros(count,dtype=np.int64)
    for name in folders:
        folder=root/name
        label=int(np.load(folder/'label.npy')); assert 0 <= label < count, folder
        distribution[label]+=1
        if require_au:
            au=np.load(folder/'au.npy')
            assert au.shape == (info['au_dim'],) and np.isfinite(au).all(), folder
            assert np.isin(au,[0,1]).all(), folder
        for suffix in (['','_gamdss'] if name in reselected else ['']):
            meta=json.loads((folder/f'motion_meta{suffix}.json').read_text())
            assert meta['schema_version'] == 1 and meta['num_classes'] == count, folder
            assert meta['motion_mode'] == info['motion_mode'], folder
            assert meta['normalization'] == 'minmax', folder
            assert meta['box_format'] == 'normalized_xyxy_exclusive', folder
            boxes=np.load(folder/f'roi_boxes{suffix}.npy')
            assert boxes.shape == (R,4) and np.isfinite(boxes).all(), folder
            assert ((boxes>=0)&(boxes<=1)).all() and (boxes[:,2:]>boxes[:,:2]).all(), folder
            perm=np.asarray(meta['roi_flip_permutation'])
            assert sorted(perm.tolist()) == list(range(R)) and np.array_equal(perm[perm],np.arange(R)), folder
            assert len(meta['roi_order']) == R, folder
            if info['motion_mode'] == 'flow':
                assert isinstance(meta['u_constant_rise'],bool) and isinstance(meta['u_constant_fall'],bool), folder
            if require_au:
                assert meta['au_flip_permutation'] == list(range(info['au_dim'])), folder
            for stem in ['inputs','onset','offset']:
                with Image.open(folder/f'{stem}{suffix}.png') as image:
                    assert image.size == (224,224), folder
            for stem in ['flow_map','flow_map_fall']:
                motion=np.load(folder/f'{stem}{suffix}.npy')
                assert motion.shape == (R,3,28,28) and np.isfinite(motion).all(), folder
                assert motion.min()>=-1e-6 and motion.max()<=1+1e-6, folder
    return dict(info, class_counts=distribution.tolist(), reselected_samples=len(reselected))
