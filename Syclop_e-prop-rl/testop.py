from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))
_tst_module = tf.load_op_library(os.path.join(dir_path, 'tfoptest.so'))

def _game_dir():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "roms")

def get_game_path(game_name):
    return os.path.join(_game_dir(), game_name) + ".bin"

def optest(action, reset, max_episode_length, game_name, **kwargs):
    return _tst_module.optest(action, reset, max_episode_length, get_game_path(game_name), **kwargs)

