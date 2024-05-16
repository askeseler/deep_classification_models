from ast import Str
from pytorch_lightning.callbacks import ProgressBar
import sys
from tqdm import tqdm
import os

class mytqdm(tqdm):
    def __init__(self,*args, **kwargs):
        kwargs["delay"] = 1e-100
        self.additional_text = None
        super().__init__(*args, **kwargs)

    def format_meter(self, n, total, elapsed, ncols=None, prefix='', ascii=False, unit='it',
                     unit_scale=False, rate=None, bar_format=None, postfix=None,
                     unit_divisor=1000, initial=0, colour=None, **extra_kwargs):
        self.additional_text = postfix
        return tqdm.format_meter(n, total, elapsed, ncols, prefix, ascii, unit,
                             unit_scale, rate, bar_format, None,#no postfix
                             unit_divisor, initial, colour, **extra_kwargs)

    def wrap(self, s, w):
        return [""] if not s else [s[i:i + w] for i in range(0, len(s), w)]
        
    def display(self, msg=None, pos=None):
        width = list(os.get_terminal_size())[0]
        lines = self.wrap(self.additional_text, width)
        lines.append("")
        self.moveto(0)
        self.sp(self.__str__())
        self.moveto(1)
        self.sp(lines[0])
        self.moveto(-1)
        self.moveto(2)
        self.sp(lines[1])
        self.moveto(-2)
        return True


class MultilineProgressBar(ProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = mytqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        )
        return bar

    def on_train_epoch_end(self, *args):
        pass

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = mytqdm(
            desc='Validating',
            position=(2 * self.process_position + 1),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = mytqdm(
            desc='Testing',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'

        )
        return bar