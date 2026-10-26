import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from models.MA3D import MA3D
from models.mobilefacenet import MobileFaceNet
from models.hyp_crossvit import HyVisionTransformer
from models.fullface_context import roi_token_coordinates, add_roi_token_args, roi_token_kwargs
from models.motion_encoder import MotionEncoderCNN

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]

class TokenModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = {'state_dict': MobileFaceNet([112,112],136).state_dict()}

    def model(self, **kwargs):
        config = dict(num_classes=3, type='small', n_roi=3, num_film_blocks=3,
                      use_rise_fall=False, rise_fall_mode='decision_level')
        config.update(kwargs)
        with patch('models.MA3D.torch.load', return_value=self.checkpoint):
            return MA3D(**config)

    def inputs(self, r):
        boxes=torch.tensor([.1,.1,.8,.9]).repeat(2,r,1)
        return (torch.randn(2,3,224,224),torch.randn(2,3,224,224),
                torch.randn(2,r,3,28,28)), boxes

    def test_configuration_matrix_forward_backward(self):
        # Both ROI protocols, both requested FiLM depths, several grid/head choices.
        for r,g,k,heads,classes,au in [(3,2,3,1,3,False),(6,3,5,4,5,True),(3,5,3,8,3,False)]:
            with self.subTest(r=r,g=g,k=k):
                model=self.model(n_roi=r,roi_pool_size=g,num_film_blocks=k,
                                 geometry_heads=heads,num_classes=classes,use_au=au)
                inputs,boxes=self.inputs(r)
                buffers={n:b.clone() for n,b in model.face_landback.named_buffers()}
                observed=[]
                handle=model.appearance_encoder.register_forward_pre_hook(lambda m,args: observed.append(args[0].shape))
                model.train()
                out,feat,attn,_=model(*inputs,x_au=torch.rand(2,36) if au else None,roi_boxes=boxes)
                torch.nn.functional.cross_entropy(out,torch.tensor([0,1])).backward()
                handle.remove()
                self.assertEqual(out.shape,(2,classes))
                self.assertEqual(observed,[torch.Size([2,3,224,224])])
                self.assertEqual(attn[0]['l']['img_attn'].shape[-2:],(r*g*g+1,50))
                self.assertEqual(attn[0]['l']['lm_attn'].shape[-2:],(50,r*g*g+1))
                for module in (model.motion_encoder,model.appearance_encoder,model.geometry_context,model.landmark_fusion):
                    grads=[p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(x).all() for x in grads))
                    self.assertGreater(sum(x.abs().sum().item() for x in grads),0)
                self.assertFalse(model.face_landback.training)
                self.assertTrue(all(p.grad is None for p in model.face_landback.parameters()))
                self.assertTrue(all(torch.equal(buffers[n],b) for n,b in model.face_landback.named_buffers()))

    def test_decision_training_and_fall_free_eval(self):
        model=self.model(n_roi=6,use_rise_fall=True)
        inputs,boxes=self.inputs(6)
        fall=torch.randn_like(inputs[2],requires_grad=True)
        out,_,_,aux=model(*inputs,x_flow_fall=fall,roi_boxes=boxes)
        self.assertIsNotNone(aux['fall'])
        (out.square().mean()+.1*aux['fall'].square().mean()).backward()
        self.assertGreater(fall.grad.abs().sum().item(),0)
        model.eval()
        with torch.no_grad():
            a=model(*inputs,roi_boxes=boxes)[0]
            b=model(*inputs,x_flow_fall=torch.full_like(fall,float('nan')),roi_boxes=boxes)[0]
        self.assertTrue(torch.equal(a,b))
        model.train()
        with self.assertRaisesRegex(ValueError,'Fall motion'):
            model(*inputs,roi_boxes=boxes)

    def test_geometry_and_position_ablations(self):
        for mode in ('cross_attention','global_mean','none'):
            with self.subTest(mode=mode):
                model=self.model(geometry_conditioning=mode,use_roi_position=False)
                inputs,boxes=self.inputs(3)
                out=model(*inputs,roi_boxes=boxes)[0]
                out.square().mean().backward()
                self.assertTrue(torch.isfinite(out).all())
                if mode=='none':
                    self.assertIsNone(model.landmark_fusion)
                else:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.landmark_fusion.parameters()))

    def test_checkpoint_roundtrip(self):
        model=self.model(roi_pool_size=2,use_roi_position=False).eval()
        other=self.model(**model.model_config).eval()
        other.load_state_dict(model.state_dict())
        inputs,boxes=self.inputs(3)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(*inputs,roi_boxes=boxes)[0],other(*inputs,roi_boxes=boxes)[0]))

    def test_feature_gate_compatibility(self):
        model=self.model(use_rise_fall=True,rise_fall_mode='feature_gate').eval()
        inputs,boxes=self.inputs(3)
        with torch.no_grad():
            out,_,_,aux=model(*inputs,x_flow_fall=inputs[2],roi_boxes=boxes)
        self.assertEqual(out.shape,(2,3))
        self.assertEqual(aux['agreement'].shape,(2,3,1,3,3))

    def test_bad_boxes_and_unsupported_configs(self):
        model=self.model()
        inputs,boxes=self.inputs(3)
        with self.assertRaises(ValueError):model(*inputs)
        boxes[...,2]=0
        with self.assertRaises(ValueError):model(*inputs,roi_boxes=boxes)
        for kw in ({'roi_pool_size':1},{'geometry_heads':3},{'num_film_blocks':0},{'motion_backbone':'rmt'}):
            with self.assertRaises(ValueError):self.model(**kw)

class InterfaceTests(unittest.TestCase):
    def test_coordinates_and_reflection(self):
        boxes=torch.tensor([[[.1,.2,.5,.8],[.5,0.,1.,1.]]])
        xy=roi_token_coordinates(boxes,2).reshape(1,2,2,2,2)
        self.assertTrue(torch.allclose(xy[0,0,0,0],torch.tensor([.2,.35])))
        flipped=boxes.clone()
        flipped[...,0]=1-boxes[...,2];flipped[...,2]=1-boxes[...,0]
        mirrored=roi_token_coordinates(flipped,2).reshape_as(xy).flip(3)
        self.assertTrue(torch.allclose(mirrored[...,0],1-xy[...,0]))
        self.assertTrue(torch.allclose(mirrored[...,1],xy[...,1]))

    def test_roi_encoder_isolation_all_grid_sizes(self):
        for g in (2,3,4,5,7):
            encoder=MotionEncoderCNN(n_roi=3,roi_pool_size=g).eval()
            x=torch.randn(1,3,3,28,28)
            with torch.no_grad():
                a=encoder.forward_token_rois(x)
                x[:,1]+=10
                b=encoder.forward_token_rois(x)
            self.assertEqual(a.shape,(1,3,512,g,g))
            self.assertTrue(torch.equal(a[:,0],b[:,0]))
            self.assertTrue(torch.equal(a[:,2],b[:,2]))

    def test_cross_interaction_legacy_equal_tokens_unchanged(self):
        spec=importlib.util.spec_from_file_location('original_cross',ROOT.parent/'original/models/hyp_crossvit.py')
        if not spec.origin or not Path(spec.origin).exists():
            self.skipTest('Original uploaded reference available in development workspace only')
        # Strip unused einops imports from the unmodified reference to avoid extra dependency.
        ns={}
        src=Path(spec.origin).read_text()
        src='\n'.join(line for line in src.splitlines() if 'einops' not in line)
        exec(compile(src,spec.origin,'exec'),ns)
        old=ns['HyVisionTransformer'](embed_dim=32,depth=1,num_heads=2).eval()
        new=HyVisionTransformer(embed_dim=32,depth=1,num_heads=2).eval()
        new.load_state_dict(old.state_dict())
        a,b=torch.randn(2,49,32),torch.randn(2,49,32)
        with torch.no_grad():self.assertTrue(torch.equal(old(a,b)[0],new(a,b)[0]))

    def test_cli_both_entrypoints_preserve_old_defaults(self):
        import typing
        for file in ('train.py','train_transfer.py'):
            tree=ast.parse((ROOT/file).read_text())
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('get_args','parse_args'))
            ns=dict(argparse=argparse,add_roi_token_args=add_roi_token_args,**vars(typing))
            exec(compile(ast.Module(body=[fn],type_ignores=[]),file,'exec'),ns)
            flags=['--n_roi','6','--num_film_blocks','5','--roi_pool_size','5',
                   '--geometry_heads','8','--use_rise_fall','--rise_fall_mode','decision_level']
            if file=='train.py':
                with patch('sys.argv',['train.py']+flags): args=ns[fn.name]()
            else:
                # transfer entrypoint has required source/target dataset arguments
                flags+=['--source','4dme','--source_root','/tmp/source','--target','casme2=/tmp/target']
                args=ns[fn.name](flags)
            self.assertEqual(args.lr,1e-5 if file=='train.py' else 5e-4)
            self.assertEqual(args.batch_size,8 if file=='train.py' else 16)
            self.assertEqual(roi_token_kwargs(args)['roi_pool_size'],5)
            self.assertTrue(args.use_rise_fall)
            if file == 'train_transfer.py':
                build = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_build_model_config')
                ns['roi_token_kwargs'] = roi_token_kwargs
                exec(compile(ast.Module(body=[build],type_ignores=[]),file,'exec'),ns)
                config = ns['_build_model_config'](args,36)
                self.assertEqual(config['roi_pool_size'],5)
                self.assertEqual(config['geometry_heads'],8)
                self.assertEqual(config['num_film_blocks'],5)

if __name__=='__main__':unittest.main()
