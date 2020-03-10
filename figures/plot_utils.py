import math
import matplotlib.pyplot as plt
import os
import subprocess
import tempfile

# PNAS figure dimensions
# 1 column: 8.7 cm
# 2 column: 17.8 cm

# Modified from https://github.com/Wookai/paper-tips-and-tricks

# https://stackoverflow.com/questions/14708695/specify-figure-size-in-centimeter-in-matplotlib
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def get_fig_size(fig_width_cm, fig_height_cm):
    """Convert dimensions in centimeters to inches"""
    size_cm = (fig_width_cm, fig_height_cm)
    return list(map(lambda x: x/2.54, size_cm))


def label_size():
    """Size of axis labels
    """
    return 8

def font_size():
    """Size of all texts shown in plots
    """
    return 6

def ticks_size():
    """Size of axes' ticks
    """
    return 8

def axis_lw():
    """Line width of the axes
    """
    return 1.25

def lines_lw():
    """Line width of the axes
    """
    return 1.5


def figure_setup(params=dict()):
    """Set all the sizes to the correct values and use
    tex fonts for all texts.
    """
    default_params = {'text.usetex': True,
              'figure.dpi': 300, #200
              'font.size': font_size(),
              'font.sans-serif': 'Arial',
              'axes.labelsize': label_size(),
              'axes.titlesize': font_size(),
              'axes.linewidth': axis_lw(),
              'lines.linewidth': lines_lw(),
              'legend.fontsize': font_size(),
              'xtick.labelsize': ticks_size(),
              'ytick.labelsize': ticks_size()}
    default_params.update(params)
    plt.rcParams.update(default_params)

def remote_display(fig, display="localhost:10.0"):
    save_fig(fig, "tmp.png", dpi=300)
    os.environ["DISPLAY"]=display
    os.system("feh tmp.png")

def save_fig(fig, file_name, fmt=None, dpi=300, tight=True):
    """Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it.
    """

    if not fmt:
        fmt = file_name.strip().split('.')[-1]

    if fmt not in ['eps', 'png', 'pdf']:
        raise ValueError('unsupported format: %s' % (fmt,))

    extension = '.%s' % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    # save figure
    if tight:
        fig.savefig(tmp_name, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(tmp_name, dpi=dpi)

    # trim it
    if fmt == 'eps':
        subprocess.call('epstool --bbox --copy %s %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'png':
        subprocess.call('convert %s -trim %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'pdf':
        subprocess.call('pdfcrop %s %s' % (tmp_name, file_name), shell=True)
