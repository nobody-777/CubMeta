import abc
import torch
import os.path as osp

from trainer_single.utils import (
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
from trainer_single.logger import Logger

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        # ensure_path(
        #     self.args.save_path,
        #     scripts_to_save=['trainer_single/models', 'trainer_single/networks', __file__],
        # )
        self.logger = Logger(args, osp.join(args.save_path))

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0
        # self.trlog = {}
        self.trlog['min_loss'] = 9999
        self.trlog['min_loss_epoch'] = 0
        self.trlog['min_loss_interval'] = 0.0

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
            vl, va, vap = self.evaluate(self.val_loader)
            self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
            self.logger.add_scalar('val_acc', float(va),  self.train_epoch)
            print('val_loss={:.4f}, val_acc={:.4f}'.format(vl,va))

            # if va >= self.trlog['max_acc']:
            #     self.trlog['max_acc'] = va
            #     self.trlog['max_acc_interval'] = vap
            #     self.trlog['max_acc_epoch'] = self.train_epoch
            #     self.save_model('max_acc')
            #     print('Best so far')
            if vl <= self.trlog['min_loss']:
                self.trlog['min_loss'] = vl
                self.trlog['min_loss_interval'] = vap
                self.trlog['min_loss_epoch'] = self.train_epoch
                self.save_model('min_loss')
                print('Best so far')

            if self.train_epoch % 5 == 0:
                self.save_model('epoch-'+ str(self.train_epoch))

    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
