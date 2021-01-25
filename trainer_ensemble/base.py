import abc
import torch
import os.path as osp

from trainer_ensemble.utils import (Averager, Timer,)
from trainer_ensemble.logger import Logger

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.val_loader = None
        self.args = args
        self.logger = Logger(args, osp.join(args.save_path))

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}

        self.trlog['min_loss'] = 9999
        self.trlog['min_loss_epoch'] = 0

        self.trlog['max_pa'] = 0
        self.trlog['max_pa_epoch'] = 0

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass
    
    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            v_l, v_pa, v_a_pa, v_va, v_a_va, v_ma, v_a_ma = self.evaluate(self.val_loader)
            self.logger.add_scalar('v_l', float(v_l), self.train_epoch)
            self.logger.add_scalar('accuracy/v_pa', float(v_pa),  self.train_epoch)
            self.logger.add_scalar('accuracy/v_va', float(v_va),  self.train_epoch)
            self.logger.add_scalar('accuracy/v_ma', float(v_ma),  self.train_epoch)

            if v_l <= self.trlog['min_loss']:
                self.trlog['min_loss'] = v_l
                self.trlog['min_loss_epoch'] = self.train_epoch
                self.save_model('min_loss')

            if v_pa >= self.trlog['max_pa']:
                self.trlog['max_pa'] = v_pa
                self.trlog['max_pa_epoch'] = self.train_epoch
                self.save_model('max_pa')

            if self.train_epoch % 5 == 0:
                self.save_model('epoch-'+ str(self.train_epoch))

    def save_model(self, name):
        torch.save(
            dict(params_a=self.models[0].state_dict(), params_b=self.models[1].state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

    def __str__(self):
        return "{}({}),{}({})".format(
            self.__class__.__name__,
            self.models[0].__class__.__name__,
            self.__class__.__name__,
            self.models[1].__class__.__name__
        )
