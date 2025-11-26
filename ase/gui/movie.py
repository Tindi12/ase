# fmt: off

import tkinter as tk

import numpy as np

import ase.gui.ui as ui
from ase.gui.i18n import _


class Movie:
    def __init__(self, gui):
        self.win = win = ui.Window(_('Movie'), close=self.close)
        win.add(_('Image number:'))
        self.frame_number = ui.Scale(gui.frame, 0,
                                     len(gui.images) - 1,
                                     callback=self.new_frame)
        win.add(self.frame_number)

        win.add([ui.Button(_('First'), self.click, -1, True),
                 ui.Button(_('Back'), self.click, -1),
                 ui.Button(_('Forward'), self.click, 1),
                 ui.Button(_('Last'), self.click, 1, True)])

        play = ui.Button(_('Play'), self.play)
        stop = ui.Button(_('Stop'), self.stop)

        # TRANSLATORS: This function plays an animation forwards and backwards
        # alternatingly, e.g. for displaying vibrational movement
        self.rock = ui.CheckButton(_('Rock'))

        win.add([play, stop, self.rock])

        if len(gui.images) > 150:
            skipdefault = len(gui.images) // 150
            tdefault = min(max(len(gui.images) / (skipdefault * 5.0),
                               1.0), 30)
        else:
            skipdefault = 0
            tdefault = min(max(len(gui.images) / 5.0, 1.0), 30)
        self.time = ui.SpinBox(tdefault, 1.0, 99, 0.1)
        self.skip = ui.SpinBox(skipdefault, 0, 99, 1)
        win.add([_(' Frame rate: '), self.time, _(' Skip frames: '),
                 self.skip])

        self.gui = gui
        self.direction = 1
        self.timer = None
        gui.obs.new_atoms.register(self.close)

    def close(self):
        self.stop()
        self.win.close()

    def click(self, step, firstlast=False):
        if firstlast and step < 0:
            i = 0
        elif firstlast:
            i = len(self.gui.images)
        else:
            i = max(0, min(len(self.gui.images) - 1, self.gui.frame + step))

        self.frame_number.value = i
        if firstlast:
            self.direction = np.sign(-step)
        else:
            self.direction = np.sign(step)

    def new_frame(self, value):
        self.gui.set_frame(value)

    def play(self):
        self.stop()
        t = 1 / self.time.value
        self.timer = self.gui.window.after(t, self.step)

    def stop(self):
        if self.timer is not None:
            self.timer.cancel()

    def step(self):
        i = self.gui.frame
        nimages = len(self.gui.images)
        delta = int(self.skip.value) + 1

        if self.rock.value:
            if i <= self.skip.value:
                self.direction = 1
            elif i >= nimages - delta:
                self.direction = -1
            i += self.direction * delta
        else:
            i = (i + self.direction * delta + nimages) % nimages

        self.frame_number.value = i
        self.play()


class MovieToolbar:
    def __init__(self, parent, gui):
        from itertools import count

        self.gui = gui

        tkframe = tk.Frame(parent, relief='raised', borderwidth=1)
        self.tkframe = tkframe

        columncounter = count()

        def add(widget):
            widget.grid(row=0, column=next(columncounter))
            return widget

        self.graphbutton_icon = tk.PhotoImage(file=gui.icons.graph).subsample(2)
        self.graphbutton = add(tk.Button(
            tkframe, image=self.graphbutton_icon,
            command=self.gui.plot_graph_standard
        ))

        self.moviebutton_icon = tk.PhotoImage(file=gui.icons.movie).subsample(2)
        self.moviebutton = add(tk.Button(
            tkframe, image=self.moviebutton_icon, command=self.gui.movie
        ))

        add(tk.Label(tkframe, text='Image:'))

        self.slider = add(
            tk.Scale(tkframe, from_=0,
                     orient='horizontal',
                     command=self.slidercommand,
                     showvalue=False)
            )
        self._update_number_of_images()
        self.numlabel = add(tk.Label(tkframe, text='0'))

        # "Set" atoms means something (anything) changed including
        # which frame number we are displaying:
        gui.obs.set_atoms.register(self._update_atoms)
        gui.obs.set_atoms.register(self._update_button_states)

        # "New" atoms may change the number of images altogether:
        gui.obs.new_atoms.register(self._update_number_of_images)

        # This is not reliable at all, for example if user opens new atoms,
        # we will not detect that.

    def _update_number_of_images(self):
        self.slider['to'] = len(self.gui.images) - 1

    def _update_atoms(self):
        self.slider.set(self.gui.frame)

    def slidercommand(self, sliderval: str):
        framenum = int(sliderval)
        self.numlabel['text'] = sliderval
        self.gui.set_frame(framenum)

    def _update_button_states(self):
        if len(self.gui.images) > 1:
            self.graphbutton['state'] = tk.ACTIVE
            self.slider['state'] = tk.ACTIVE
        else:
            self.graphbutton['state'] = tk.DISABLED
            self.slider['state'] = tk.DISABLED
