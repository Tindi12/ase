
import os.path as op


class Icons:
    def __init__(self):
        icon_dir = op.join(op.dirname(op.realpath(__file__)), 'icons')
        self.play = op.join(icon_dir, 'CarbonPlay.png')
        self.pause = op.join(icon_dir, 'CarbonPause.png')
        self.movie = op.join(icon_dir, 'CarbonDataPlayer.png')
        self.graph = op.join(icon_dir, 'CarbonChartLine.png')
