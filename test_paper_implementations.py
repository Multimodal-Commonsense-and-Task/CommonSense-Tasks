"""Run: python -m unittest test_paper_implementations -v"""
import importlib.util
from pathlib import Path
import unittest
import torch


def module(folder):
    spec=importlib.util.spec_from_file_location(folder,Path(__file__).parent/folder/'model.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class PaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_ma_filter_backpropagates_to_boundaries(self):
        m=module('MA-VR'); net=m.MA_VR(8,8,16,4)
        _,_,r=net(torch.randn(2,6,8),torch.randn(2,3,8)); r.sum().backward()
        self.assertGreater(net.boundary.weight.grad.abs().sum().item(),0)

    def test_ma_span_validity_and_negative_loss(self):
        m=module('MA-VR'); scores=m.corpus_scores(torch.zeros(2,4),torch.zeros(2,4),torch.zeros(2))
        self.assertTrue(torch.isneginf(scores[:,3,0]).all())
        self.assertTrue(torch.isfinite(scores[:,0,3]).all())
        x=torch.randn(2,4,requires_grad=True)
        loss=m.ranking_loss(x,x,torch.tensor([0,0])); loss.backward(); self.assertEqual(loss.item(),0)

    def test_obd_has_no_future_dependency(self):
        m=module('ESTimator'); model=m.CEA(16,4,1,4).eval()
        x=torch.randn(15,16); changed=x.clone(); changed[10:]*=100
        before=m.stream(model,x); after=m.stream(model,changed)
        self.assertEqual(before[0][:10],after[0][:10]); self.assertEqual(before[1][:10],after[1][:10])
        obd=m.OBD(3); [obd.update(v) for v in [1.,1.,1.]]
        self.assertTrue(obd.update(3.)); self.assertEqual(obd.queue[-1],3.)

    def test_losses_and_idi(self):
        m=module('IDI'); z=torch.randn(6,8,requires_grad=True)
        loss=m.supcon(z,torch.tensor([0,0,1,1,2,2])); loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertAlmostEqual(m.idi([.8,.9],[.2,.3],[.8,.9]),1.)
        self.assertAlmostEqual(m.idi([.8,.9],[.2,.3],[.2,.3]),0.)
        self.assertLess(m.idi([.8,.9],[.2,.3],[.1,.1]),0.)
        self.assertIsNone(m.idi([.2],[.2],[.1]))
        self.assertAlmostEqual(m.nce_bound(torch.zeros(8,8)).item(),0.,places=5)


if __name__=='__main__': unittest.main()
