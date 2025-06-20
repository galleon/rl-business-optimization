import unittest.mock
import matplotlib.pyplot as plt

_original_plt_show_func = None

def patch_plt_show():
    global _original_plt_show_func
    if _original_plt_show_func is None:
        _original_plt_show_func = plt.show
        plt.show = unittest.mock.MagicMock()

def unpatch_plt_show():
    global _original_plt_show_func
    if _original_plt_show_func is not None:
        plt.show = _original_plt_show_func
        # Optional: _original_plt_show_func = None
