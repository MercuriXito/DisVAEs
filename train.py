"""
Trainer of VAE
"""
import os, time
from tqdm import tqdm

import warnings
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import get_loader, get_utiler
from opt import get_config, get_model
from utils import test_and_add_postfix_dir, test_and_make_dir, \
        json_dump, TensorImageUtils, currentTime


class VAETrainer:
    def __init__(self, 
            opt,
            vae,
            data,
            device=None,
            testing=False, # TODO: test mode
            **kws
        ):

        self.vae = vae
        self.data = data
        self.data_size = len(self.data.dataset)
        self.opt = opt
        self.testing= testing

        # pre-defined save root
        save_root = test_and_add_postfix_dir("logs" + os.sep + "save_" + currentTime())
        test_and_make_dir(save_root)
        print("Saving results at {}".format(save_root))
        self.base_root = save_root
        self._create_dir()
        self.utiler = get_utiler(opt.data_name, self.img_root)

        # opt
        self.cuda = self.opt.use_cuda
        self.lr = self.opt.lr
        self.betas = self.opt.betas
        self.lr_decay_weight = self.opt.lr_decay_weight
        self.decay_step = self.opt.decay_step
        self.nrow = self.opt.grid_nrow
        self.epochs = self.opt.epochs
        self.num_epoch_save = self.opt.num_epoch_save

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")

        # logger 
        self.writer = SummaryWriter(self.base_root)
        self.save_config()


    def _init_train(self):
        """ initialize include:
            + resume model if specified, or initialize model
            + create relevant 
        """
        self.epoch_step = 0
        self.iter_step = 0

        self.vae.to(self.device)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=self.lr, betas=tuple(self.betas))
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.decay_step,
                gamma=self.lr_decay_weight)

        self._resume_train()

    def _resume_train(self):
        """ resume model and optimizer """
        if not self.opt.resume:
            return
        folder = test_and_add_postfix_dir(self.opt.resume_folder) + "checkpoints" + os.sep

        model_path = folder + "vae.pt"
        optim_path = folder + "optim.pt"
        scheduler_path = folder + "lr_scheduler.pt"

        names = ["model", "optimizer", "lr_scheduler"]
        load_objs = [ self.vae, self.optimizer, self.lr_scheduler ]
        paths = [ model_path, optim_path, scheduler_path ]

        for name, load_obj, path in zip(names, load_objs, paths):
            if os.path.exists(path):
                print("Loading resuemed {} in:{}".format(name, path))
                load_obj.load_state_dict(torch.load(path))
            else:
                warnings.warn("No detected {} of \"{}\", skip resume {}".format(name, path, name))

    def train_in_one_iter(self):

        images, _ = self.batches
        size = images.size(0)
        images = images.type(torch.float32).to(self.device)

        # update
        self.optimizer.zero_grad()
        reconst, mu, logvar, z = self.vae(images)
        losses, additional_records = self.vae.get_loss(images, reconst, mu, logvar, z, self.iter_step)
        total_loss = losses["total_loss"]
        total_loss.backward()
        self.optimizer.step()

        # log
        for name, loss in losses.items():
            self.writer.add_scalar(name, loss.item(), self.iter_step)
        self.bar.set_description(
            " - ".join(["[%s: %2.6f]" % (name, loss.item()) for name, loss in losses.items()])
            )

        # log additional records from model
        for name, record in additional_records.items():
            self.writer.add_scalar(name, record, self.iter_step)

    def log_in_iter(self): # Not used right now
        pass

    def log_in_epoch(self):
        # save model
        self.save_models(self.vae, "vae.pt")
        self.save_models(self.optimizer, "optim.pt")
        self.save_models(self.lr_scheduler, "lr_scheduler.pt")
        # save sample images and reconst images
        with torch.no_grad():
            images, _ = self.batches
            images = images.type(torch.float32).to(self.device)
            reconst = self.vae(images)[0]
            sample_images = self.vae.infer(reconst.size(0))
        self.save_images(reconst, "reconst_{}.png".format(self.epoch_step + 1), nrow=self.nrow)
        self.save_images(sample_images, "sample_{}.png".format(self.epoch_step + 1), nrow=self.nrow)


    def run(self):
        self._init_train()
        print("Start Training.")
        starttime = time.clock()
        for epoch in range(self.epochs):
            print("Epoch:{}/{}".format(epoch+1, self.epochs))
            self.bar = tqdm(self.data)
            for i, batches in enumerate(self.bar):
                self.batches = batches
                self.train_in_one_iter()
                # if self.num_iter_save > 0 and i % self.num_iter_save == 0:
                #     self.log_in_iter()
                self.iter_step += 1
            self.lr_scheduler.step()
            if self.epoch_step % self.num_epoch_save == 0 or self.epoch_step == epochs -1:
                self.log_in_epoch()
            self.epoch_step += 1
        endtime = time.clock()
        consume_time = endtime - starttime
        print("Training Complete, Using %d min %d s" %(consume_time // 60,consume_time % 60))

    def validate(self):
        """ validate metrics """
        pass

    def save_config(self):
        json_dump(self.opt._get_kwargs(), self.base_root + "config.json")

    def save_images(self, images, name, **kws):
        self.utiler.save_images(images, name, **kws)

    def save_models(self, net, name):
        torch.save(net.state_dict(), self.model_root + name)

    def save_checkpoints(self, obj, name):
        torch.save(obj, self.model_root + name)

    def _create_dir(self):
        """
        + base_root:
            + checkpoint: models
            + images: 
            + config file
        """
        self.img_root = self.base_root + "images" + os.sep
        self.model_root = self.base_root + "checkpoints" + os.sep

        for r in [self.img_root, self.model_root]:
            test_and_make_dir(r)

def main():

    opt = get_config()
    print("Using configs:\n{}".format(opt))
    vae = get_model(opt)
    print("Using model:\n{}".format(vae))

    data = get_loader(opt)
    trainer = VAETrainer(
            opt, vae, data,
            )
    trainer.run()

if __name__ == '__main__':
    main()
