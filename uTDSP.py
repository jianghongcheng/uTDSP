'''
The code is based on the implementation provided in https://github.com/liuofficial/SDP.
'''

import time
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from data.data_info import DataInfo
from utils.torchkits import torchkits
from utils.toolkits import toolkits
from utils.blur_down import BlurDown
from utils.ema import EMA
from model.uTDSP_net import NetConfig, Activation
from model.gaussian_diffusion import GaussianDiffusion
from blind import Blind
from metrics import psnr_loss, ssim, ergas, cc
from metrics import sam as sam_t




class Net(nn.Module):
    def __init__(self, hs_bands, layers=4, timesteps=1000):
        super().__init__()
        self.hs_bands = hs_bands
        self.timesteps = timesteps

        self.net = NetConfig(
            num_channels=self.hs_bands,
            skip_layers=tuple(range(1, layers)),
            num_hid_channels=512,
            num_layers=layers,
            num_time_emb_channels=64,
            activation=Activation.silu,
            use_norm=True,
            condition_bias=1.0,
            dropout=0.001,
            last_act=Activation.none,
            num_time_layers=2,
            time_last_act=False
        ).make_model()

        self.gauss_diffusion = GaussianDiffusion(denoise_fn=self.net, timesteps=self.timesteps, improved=False)
        self._init_weights()

    def forward(self, X):
        return self.gauss_diffusion.train_losses(X)

    def cpt_loss(self, output, label=None):
        return output

    def sample(self, batch_size, device, continuous=False, idx=None):
        shape = (batch_size, self.hs_bands)
        return self.gauss_diffusion.sample(shape=shape, device=device, continuous=continuous, idx=idx)

    def _init_weights(self, init_type='normal'):
        for _, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if init_type == 'normal':
                    nn.init.xavier_normal_(m.weight.data)
                elif init_type == 'uniform':
                    nn.init.xavier_uniform_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)


#  Training
class uTDSP(DataInfo):
    def __init__(self, ndata, nratio=8, nsnr=0):
        super().__init__(ndata, nratio, nsnr)
        lr = [1e-2, 1e-2, 1e-2, 0.5e-2]
        self.lr = lr[ndata]
        self.lr_fun = lambda epoch: 0.001 * max(1000 - epoch / 10, 1)
        layers = [5, 5, 5, 5]
        self.model = Net(self.hs_bands, layers=layers[ndata])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_fun)
        torchkits.get_param_num(self.model)
        toolkits.check_dir(self.model_save_path)

        for name, parameters in self.model.named_parameters():
            print(name, ':', parameters.size())
        print(self.model)

        # FLOPs Estimation
        self.estimate_flops()

        self.model_save_pkl = self.model_save_path + 'spec.pkl'
        self.model_save_time = self.model_save_path + 't.mat'

    def estimate_flops(self):
        try:
            from fvcore.nn import FlopCountAnalysis, parameter_count_table

            # ✅ Wrap model and timestep in a Module
            class Wrapper(nn.Module):
                def __init__(self, denoise_fn, timestep):
                    super().__init__()
                    self.denoise_fn = denoise_fn
                    self.timestep = timestep

                def forward(self, x):
                    return self.denoise_fn(x, self.timestep)

            # Dummy input [1, Bands]
            dummy_input = torch.randn(1, self.hs_bands).cuda()
            dummy_t = torch.zeros(1, dtype=torch.long).cuda()

            # Wrap the network with the timestep
            wrapped_model = Wrapper(self.model.net.cuda(), dummy_t)

            flops = FlopCountAnalysis(wrapped_model, dummy_input)
            print(f"\n🧮 Estimated FLOPs per denoising step: {flops.total() / 1e6:.2f} MFLOPs")
            print(parameter_count_table(self.model.net))

            total_flops = flops.total() * self.model.timesteps
            print(f"🧮 Total Estimated FLOPs for diffusion (× {self.model.timesteps} steps): {total_flops / 1e9:.2f} GFLOPs")

        except ImportError:
            print("⚠️ Please install fvcore to enable FLOPs estimation.")
        except Exception as e:
            print("⚠️ FLOPs estimation failed:", e)


    def convert_data(self, img):
        _, B, H, W = img.shape
        return img.reshape(B, H * W).permute(1, 0)

    def train(self, max_iter=30000, batch_size=1024):
        cudnn.benchmark = True
        fed_data = self.convert_data(torch.tensor(self.tgt)).cuda()
        model = self.model.cuda()
        model.train()
        ema = EMA(model, 0.999)
        ema.register()
        time_start = time.perf_counter()

        for epoch in range(max_iter):
            lr = self.optimizer.param_groups[0]['lr']
            t = np.random.randint(0, fed_data.shape[0], size=(batch_size,))
            output = self.model(fed_data[t, :])
            loss = model.cpt_loss(output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            ema.update()

            if epoch % 1000 == 0:
                tol = torchkits.to_numpy(loss)
                print(epoch, lr, tol)
                torch.save(self.model.state_dict(), self.model_save_pkl)
            if epoch == max_iter - 1:
                ema.apply_shadow()
                torch.save(ema.model.state_dict(), self.model_save_pkl)

        run_time = time.perf_counter() - time_start
        sio.savemat(self.model_save_time, {'t': run_time})

    def show(self):
        model = self.model.cuda()
        model.eval()
        model.load_state_dict(torch.load(self.model_save_pkl))
        gen_spec = model.sample(10, device='cuda', continuous=False)
        spec = torchkits.to_numpy(gen_spec)
        plt.figure(num=0)
        plt.plot(spec.T)
        plt.show()


class Target(nn.Module):
    def __init__(self, hs_bands, height, width):
        super(Target, self).__init__()
        self.height = height
        self.width = width
        self.img = nn.Parameter(torch.ones(1, hs_bands, height, width))
        self.img.requires_grad = True

    def get_image(self):
        return self.img

    def check(self):
        self.img.data.clamp_(0.0, 1.0)


class TDSP(DataInfo):
    def __init__(self, ndata, nratio=8, nsnr=0, psf=None, srf=None):
        super().__init__(ndata, nratio, nsnr)
        self.strX = 'X.mat'
        if psf is not None:
            self.psf = psf
        if srf is not None:
            self.srf = srf

        self.spec_net = uTDSP(ndata, nratio, nsnr)
        lrs = [1e-3, 1e-3, 2.5e-3, 8e-3]
        self.lr = lrs[ndata]
        self.ker_size = self.psf.shape[0]
        lams = [0.1, 0.1, 0.1, 1.0]
        self.lam_A, self.lam_B, self.lam_C = lams[ndata], 1, 1e-6
        self.lr_fun = lambda epoch: 1.0

        self.psf = torch.tensor(np.reshape(self.psf, (1, 1, self.ker_size, self.ker_size)))
        self.srf = torch.tensor(np.reshape(self.srf, (self.ms_bands, self.hs_bands, 1, 1)))
        self.__hsi = torch.tensor(self.hsi)
        self.__msi = torch.tensor(self.msi)
        toolkits.check_dir(self.model_save_path)
        self.model_save_pkl = self.model_save_path + 'prior.pkl'
        self.blur_down = BlurDown()

    def cpt_loss(self, X, hsi, msi, psf, srf):
        Y = self.blur_down(X, psf, int((self.ker_size - 1) / 2), self.hs_bands, self.ratio)
        Z = func.conv2d(X, srf, None)
        return self.lam_A * func.mse_loss(Y, hsi, reduction='sum') + self.lam_B * func.mse_loss(Z, msi, reduction='sum')

    def img_to_spec(self, X):
        return X.reshape(self.hs_bands, -1).permute(1, 0)

    def spec_to_img(self, X):
        return X.reshape(1, self.height, self.width, self.hs_bands).permute(0, 3, 1, 2)

    def train(self, gam=1e-3):
        cudnn.benchmark = True
        model = Target(self.hs_bands, self.height, self.width).cuda()
        opt = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.lam_C)
        scheduler = optim.lr_scheduler.LambdaLR(opt, self.lr_fun)
        torchkits.get_param_num(model)

        hsi = self.__hsi.cuda()
        msi = self.__msi.cuda()
        psf = self.psf.cuda()
        srf = self.srf.cuda()

        self.spec_net.model.load_state_dict(torch.load(self.spec_net.model_save_pkl))
        self.spec_net.model.to(device=msi.device)
        self.spec_net.model.eval()
        for param in self.spec_net.model.parameters():
            param.requires_grad = False

        timesteps = self.spec_net.model.timesteps
        model.train()
        ema = EMA(model, 0.9)
        ema.register()
        time_start = time.perf_counter()

        for i in range(timesteps):
            lr = opt.param_groups[0]['lr']
            t = torch.full((self.height * self.width,), timesteps - 1 - i, device=msi.device, dtype=torch.long)

            for _ in range(3):
                img = model.get_image()
                spec = self.img_to_spec(img)
                noise = torch.randn_like(spec)
                xt = self.spec_net.model.gauss_diffusion.q_sample(spec, t, noise=noise)
                noise_pred = self.spec_net.model.gauss_diffusion.denoise_fn(xt, t)

                spat_spec_loss = self.cpt_loss(img, hsi, msi, psf, srf)
                spec_prior_loss = func.mse_loss(noise_pred, noise, reduction='sum')
                loss = spat_spec_loss + gam * spec_prior_loss

                opt.zero_grad()
                loss.backward()
                opt.step()
                model.check()
                ema.update()

            scheduler.step()

            if i % 100 == 0 and self.ref is not None:
                img = model.get_image()
                psnr = toolkits.psnr_fun(self.ref, torchkits.to_numpy(img))
                sam = toolkits.sam_fun(self.ref, torchkits.to_numpy(img))
                print(i, psnr, sam, loss.data, lr)

        run_time = time.perf_counter() - time_start
        ema.apply_shadow()
        img = torchkits.to_numpy(ema.model.get_image())
        img_tensor = torch.from_numpy(img)
        ref_tensor = torch.from_numpy(self.ref)

        SSIM = ssim(img_tensor, ref_tensor, 11, 'mean', 1.)
        SAM = sam_t(img_tensor, ref_tensor)
        EARGAS = ergas(img_tensor, ref_tensor)
        PSNR = psnr_loss(img_tensor, ref_tensor, 1.)
        CC = cc(img_tensor, ref_tensor)

        print(f"PSNR: {PSNR}  SSIM: {SSIM}  SAM: {SAM}  ERGAS: {EARGAS}  CC: {CC}")
        out_mat = img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        sio.savemat(self.save_path + self.strX, {'X': out_mat})


if __name__ == '__main__':
    ndata, nratio, nsnr = 0, 8, 0

    spec_net = uTDSP(ndata=ndata, nratio=nratio, nsnr=nsnr)
    spec_net.train()

    blind = Blind(ndata=ndata, nratio=nratio, nsnr=nsnr, blind=True, kernel=8)
    blind.train()
    blind.get_save_result(is_save=True)

    gams = [1e-3, 1e-3, 1e-3, 1e-1]
    net = TDSP(ndata=ndata, nratio=nratio, nsnr=nsnr, psf=blind.psf, srf=blind.srf)
    net.train(gam=gams[ndata])
