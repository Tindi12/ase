import os.path as op
import tkinter as tk


class Icons:
    def __init__(self):
        icon_dir = op.join(op.dirname(op.realpath(__file__)), 'icons')
        self.files = {}
        self.files['play'] = op.join(icon_dir, 'CarbonPlay.png')
        self.files['pause'] = op.join(icon_dir, 'CarbonPause.png')
        self.files['movie'] = op.join(icon_dir, 'CarbonDataPlayer.png')
        self.files['graph'] = op.join(icon_dir, 'CarbonChartLine.png')

        self.images = {}

        self.scale = 3.0

    def create_photoimages(self):
        """Creates PhotoImage objects that Tkinter can use. This must be done
        after the main window is created."""
        # Pillow is super convenient here, but we perhaps can't assume
        # people have it installed
        try:
            from PIL import Image, ImageTk
        except ModuleNotFoundError:
            ImageTk = None

        for key in self.files:
            if ImageTk is None:
                self.images[key] = tk.PhotoImage(
                    file=self.files[key]
                ).subsample(2)
            else:
                img = Image.open(fp=self.files[key])
                dim = round(self.scale * 20)
                self.images[key] = ImageTk.PhotoImage(img.resize([dim, dim]))

    def set_scaling(self, scale):
        self.scale = scale

    def __getitem__(self, key):
        if not self.images:
            self.create_photoimages()
        return self.images[key]
