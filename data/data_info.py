import numpy as np
import scipy.io as sio
from utils.toolkits import toolkits


class DataInfo:

    def __init__(self, ndata=0, nratio=8, nsnr=0):
        name = self.__class__.__name__
        print('%s is running' % name)
        self.gen_path = ''  # change
        self.folder_names = ['data/']
        self.data_names = ['PaviaU_256_']
        self.noise = ['_4']
        self.file_path = self.gen_path + self.folder_names[ndata] + self.data_names[ndata] + str(nratio) + self.noise[
            nsnr] + '.mat'
        print(self.gen_path + self.folder_names[ndata] + self.data_names[ndata] + str(nratio) + self.noise[nsnr])
        mat = sio.loadmat(self.file_path)
        hsi, msi = mat['LRMS'], mat['PAN']  # h x w x L, H x W x l
        msi = np.expand_dims(msi, axis=-1)  # Reshape to (256, 256, 1)
        ref = mat['HRMS'] 
        tgt = mat['LRMS']

        if 'K' in mat.keys():
            psf, srf = mat['K'], mat['R']  # K x K, l X L
        else:
            psf = np.ones(shape=(msi.shape[0] // hsi.shape[0], msi.shape[1] // hsi.shape[1]))
            srf = np.ones(shape=(msi.shape[-1], hsi.shape[-1]))
        self.save_path = self.gen_path + self.folder_names[ndata] + name + '/t1000' + str(self.data_names) + self.noise[
            nsnr] + '/'
        hsi = hsi.astype(np.float32)
        msi = msi.astype(np.float32)
        ref = ref.astype(np.float32)
        tgt = tgt.astype(np.float32)
        self.psf = psf.astype(np.float32)
        self.srf = srf.astype(np.float32)
        self.model_save_path = self.save_path + 'model/'
        # preprocess
        self.hsi = toolkits.channel_first(hsi)  # 1 x L x h x w
        self.msi = toolkits.channel_first(msi)  # 1 x l x H x W
        self.ref = toolkits.channel_first(ref)  # 1 x l x H x W
        self.tgt = toolkits.channel_first(tgt)  # 1 x l x H x W
        self.hs_bands, self.ms_bands = self.hsi.shape[1], self.msi.shape[1]
        self.ratio = int(self.msi.shape[-1] / self.hsi.shape[-1])
        self.height, self.width = self.msi.shape[2], self.msi.shape[3]

        pass
