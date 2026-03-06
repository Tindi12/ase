# fmt: off

"""Structure optimization dialog for the ASE GUI.

Provides a window where the user can choose a calculator, optimizer,
convergence parameters, output files, and then run a structure relaxation
in a background thread — keeping the GUI fully responsive.
"""

import queue
import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import asksaveasfilename

import ase.gui.ui as ui
from ase.gui.i18n import _

# ---------------------------------------------------------------------------
# Available calculators
# Keys are display names; values are factory callables (imported lazily).
# ---------------------------------------------------------------------------
_CALCULATOR_NAMES = [
    'EMT',
    'MACE-MP',
    'ORB-v3 (conservative)',
    'LennardJones',
]

# ---------------------------------------------------------------------------
# Available optimizers
# ---------------------------------------------------------------------------
_OPTIMIZER_NAMES = [
    'BFGS',
    'FIRE',
    'LBFGS',
    'FIRE2',
    'MDMin',
]


def _make_calculator(name: str):
    """Import and return a calculator instance for *name*.

    All imports are deferred so missing optional packages only raise an error
    when the user actually tries to use that calculator.
    """
    if name == 'EMT':
        from ase.calculators.emt import EMT
        return EMT()

    elif name == 'MACE-MP':
        try:
            from mace.calculators import mace_mp
        except ImportError as exc:
            raise ImportError(
                'MACE-MP is not installed.\n'
                'Install it with:  pip install mace-torch'
            ) from exc
        return mace_mp(model='medium', dispersion=False,
                       default_dtype='float32', device='cuda')

    elif name == 'ORB-v3 (conservative)':
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
        except ImportError as exc:
            raise ImportError(
                'orb-models is not installed.\n'
                'Install it with:  pip install orb-models'
            ) from exc
        orbff = pretrained.orb_v3_conservative_inf_omat(
            device='cuda',
            precision='float32-high',
        )
        return ORBCalculator(orbff, device='cuda')

    elif name == 'LennardJones':
        from ase.calculators.lj import LennardJones
        return LennardJones()

    else:
        raise ValueError(f'Unknown calculator: {name!r}')


def _make_optimizer(name: str, atoms, logfile, trajectory, **extra_kwargs):
    """Return an Optimizer instance for *atoms*."""
    kwargs = dict(logfile=logfile or None,
                  trajectory=trajectory or None)
    kwargs.update(extra_kwargs)

    if name == 'BFGS':
        from ase.optimize import BFGS
        return BFGS(atoms, **kwargs)
    elif name == 'FIRE':
        from ase.optimize import FIRE
        return FIRE(atoms, **kwargs)
    elif name == 'LBFGS':
        from ase.optimize import LBFGS
        return LBFGS(atoms, **kwargs)
    elif name == 'FIRE2':
        from ase.optimize.fire2 import FIRE2
        return FIRE2(atoms, **kwargs)
    elif name == 'MDMin':
        from ase.optimize import MDMin
        return MDMin(atoms, **kwargs)
    else:
        raise ValueError(f'Unknown optimizer: {name!r}')


# ---------------------------------------------------------------------------
# Main dialog class
# ---------------------------------------------------------------------------

class OptimizationWindow:
    """Dialog window for running a structure optimization from the ASE GUI."""

    def __init__(self, gui):
        self.gui = gui
        self._original_atoms = gui.atoms.copy()
        self._thread = None
        self._queue = queue.Queue()
        self._poll_id = None  # tk.after() handle
        self._opt_params = {}

        # ---- Build the top-level window ------------------------------------
        self.win = tk.Toplevel()
        self.win.title(_('Optimize Structure'))
        self.win.resizable(False, False)
        self.win.protocol('WM_DELETE_WINDOW', self._on_close)

        pad = dict(padx=8, pady=4)

        outer = tk.Frame(self.win, padx=12, pady=10)
        outer.pack(fill=tk.BOTH, expand=True)

        # ---- Grid of labeled controls -------------------------------------
        grid = tk.Frame(outer)
        grid.pack(fill=tk.X)

        row = 0

        # Calculator
        tk.Label(grid, text=_('Calculator:'), anchor='w')\
            .grid(row=row, column=0, sticky='w', **pad)
        self._calc_var = tk.StringVar(value=_CALCULATOR_NAMES[0])
        calc_cb = ttk.Combobox(grid, textvariable=self._calc_var,
                               values=_CALCULATOR_NAMES,
                               state='readonly', width=30)
        calc_cb.grid(row=row, column=1, columnspan=2, sticky='ew', **pad)
        row += 1

        # Optimizer
        tk.Label(grid, text=_('Optimizer:'), anchor='w')\
            .grid(row=row, column=0, sticky='w', **pad)
        self._opt_var = tk.StringVar(value=_OPTIMIZER_NAMES[0])
        opt_cb = ttk.Combobox(grid, textvariable=self._opt_var,
                              values=_OPTIMIZER_NAMES,
                              state='readonly', width=30)
        opt_cb.grid(row=row, column=1, columnspan=2, sticky='ew', **pad)
        opt_cb.bind('<<ComboboxSelected>>',
                    lambda e: self._on_optimizer_changed())
        row += 1

        # Dynamic optimizer parameter frame (rebuilt on each dropdown change)
        self._opt_params_frame = tk.LabelFrame(
            grid, text='Optimizer Parameters', padx=4, pady=4)
        self._opt_params_frame.grid(
            row=row, column=0, columnspan=3, sticky='ew', padx=8, pady=4)
        row += 1

        # Maximum Force
        tk.Label(grid, text=_('Maximum Force:'), anchor='w')\
            .grid(row=row, column=0, sticky='w', **pad)
        self._fmax_var = tk.StringVar(value='0.05')
        tk.Entry(grid, textvariable=self._fmax_var, width=15)\
            .grid(row=row, column=1, columnspan=2, sticky='w', **pad)
        row += 1

        # Maximum Steps
        tk.Label(grid, text=_('Maximum Steps:'), anchor='w')\
            .grid(row=row, column=0, sticky='w', **pad)
        self._steps_var = tk.StringVar(value='500')
        tk.Entry(grid, textvariable=self._steps_var, width=15)\
            .grid(row=row, column=1, columnspan=2, sticky='w', **pad)
        row += 1

        # Logfile
        tk.Label(grid, text=_('Logfile:'), anchor='w')\
            .grid(row=row, column=0, sticky='w', **pad)
        self._log_var = tk.StringVar(value='')
        tk.Entry(grid, textvariable=self._log_var, width=25)\
            .grid(row=row, column=1, sticky='ew', **pad)
        tk.Button(grid, text=_('Save File Dialog'),
                  command=self._browse_log)\
            .grid(row=row, column=2, sticky='w', **pad)
        row += 1

        # Trajectory
        tk.Label(grid, text=_('Trajectory:'), anchor='w')\
            .grid(row=row, column=0, sticky='w', **pad)
        self._traj_var = tk.StringVar(value='')
        tk.Entry(grid, textvariable=self._traj_var, width=25)\
            .grid(row=row, column=1, sticky='ew', **pad)
        tk.Button(grid, text=_('Save File Dialog'),
                  command=self._browse_traj)\
            .grid(row=row, column=2, sticky='w', **pad)
        row += 1

        grid.columnconfigure(1, weight=1)

        # ---- Run button + progress bar ------------------------------------
        bottom = tk.Frame(outer)
        bottom.pack(fill=tk.X, pady=(8, 0))

        self._run_btn = tk.Button(bottom, text=_('Run'),
                                  command=self._on_run,
                                  width=12, bg='#e0e0e0',
                                  relief=tk.RAISED)
        self._run_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._reset_btn = tk.Button(bottom, text='Reset Atoms',
                            command=self._on_reset, width=12, bg='#ffe0e0')
        self._reset_btn.pack(side=tk.LEFT, padx=(0, 8))

        progress_frame = tk.Frame(bottom)
        progress_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._progress = ttk.Progressbar(progress_frame,
                                         orient=tk.HORIZONTAL,
                                         mode='indeterminate',
                                         length=200)
        self._progress.pack(fill=tk.X, expand=True)

        # Colored status strip below the bar (green done / red error)
        self._status_canvas = tk.Canvas(progress_frame, height=6,
                                        bg='#d0d0d0', highlightthickness=0)
        self._status_canvas.pack(fill=tk.X, expand=True, pady=(2, 0))

        # ---- Status label -------------------------------------------------
        self._status_var = tk.StringVar(value=_('Ready.'))
        tk.Label(outer, textvariable=self._status_var,
                 anchor='w', fg='#444444')\
            .pack(fill=tk.X, pady=(4, 0))

        self._set_running(False)
        # Schedule param panel build after window is fully rendered
        self.win.after(1, self._on_optimizer_changed)

    # -----------------------------------------------------------------------
    # File choosers
    # -----------------------------------------------------------------------

    def _browse_log(self):
        path = asksaveasfilename(
            title=_('Choose logfile'),
            defaultextension='.log',
            filetypes=[(_('Log files'), '*.log'), (_('All files'), '*')],
        )
        if path:
            self._log_var.set(path)

    def _browse_traj(self):
        path = asksaveasfilename(
            title=_('Choose trajectory file'),
            defaultextension='.traj',
            filetypes=[(_('ASE Trajectory'), '*.traj'), (_('All files'), '*')],
        )
        if path:
            self._traj_var.set(path)

    # -----------------------------------------------------------------------
    # UI state helpers
    # -----------------------------------------------------------------------

    def _set_running(self, running: bool):
        state = 'disabled' if running else 'normal'
        self._run_btn.config(state=state)

    def _set_status(self, text: str, color: str = '#444444'):
        self._status_var.set(text)
        # Can't set label fg with StringVar easily, so update widget directly
        for widget in self.win.winfo_children():
            pass  # label is inside outer frame; done by direct config below

    def _set_status_strip(self, color: str):
        """Color the thin strip under the progress bar."""
        self._status_canvas.config(bg=color)

    def _on_reset(self):
        self.gui.new_atoms(self._original_atoms.copy())
        self.gui.draw()
        self._progress.config(value=0)
        self._status_var.set('Atoms reset to original state.')
        self._set_status_strip('#d0d0d0')

    def _on_optimizer_changed(self):
        """Rebuild the parameter panel when the optimizer dropdown changes."""
        # Remove all widgets currently inside the frame
        for widget in self._opt_params_frame.winfo_children():
            widget.destroy()
        # Clear the stored variable dict
        self._opt_params = {}

        name = self._opt_var.get()
        if name == 'BFGS':
            self._add_param('maxstep', '0.2', 'Max step size (\u00c5)')
            self._add_param('alpha', '70.0', 'Initial Hessian estimate')
        elif name == 'FIRE':
            self._add_param('dt', '0.1', 'Initial time step (fs)')
            self._add_param('dtmax', '1.0', 'Max time step (fs)')
            self._add_param('maxstep', '0.2', 'Max step size (\u00c5)')
            self._add_param('Nmin', '5', 'Min steps before acceleration')
            self._add_param('finc', '1.1', 'Time step increase factor')
            self._add_param('fdec', '0.5', 'Time step decrease factor')
            self._add_param('astart', '0.1', 'Velocity mixing (start)')
        elif name == 'LBFGS':
            self._add_param('maxstep', '0.2', 'Max step size (\u00c5)')
            self._add_param('memory', '100', 'Steps to remember')
        elif name == 'FIRE2':
            self._add_param('dt', '0.1', 'Initial time step (fs)')
            self._add_param('dtmax', '1.0', 'Max time step (fs)')
            self._add_param('maxstep', '0.2', 'Max step size (\u00c5)')
        elif name == 'MDMin':
            self._add_param('dt', '0.02', 'Time step (fs)')
        # Resize window to fit new content
        self.win.update_idletasks()
        self.win.geometry('')

    def _add_param(self, name, default, label):
        """Add one labeled Entry field to the optimizer parameter frame."""
        row = len(self._opt_params_frame.winfo_children()) // 2
        var = tk.StringVar(value=default)
        self._opt_params[name] = var
        tk.Label(
            self._opt_params_frame,
            text=label + ':',
            anchor='w',
        ).grid(row=row, column=0, sticky='w', padx=4, pady=2)
        tk.Entry(
            self._opt_params_frame,
            textvariable=var,
            width=10,
        ).grid(row=row, column=1, sticky='w', padx=4, pady=2)

    # -----------------------------------------------------------------------
    # Run logic
    # -----------------------------------------------------------------------

    def _on_run(self):
        if self._thread is not None and self._thread.is_alive():
            return  # already running

        # Validate inputs
        try:
            fmax = float(self._fmax_var.get())
        except ValueError:
            ui.error(_('Invalid input'), _('Maximum Force must be a number.'))
            return

        try:
            steps = int(self._steps_var.get())
        except ValueError:
            ui.error(_('Invalid input'), _('Maximum Steps must be an integer.'))
            return

        calc_name = self._calc_var.get()
        opt_name = self._opt_var.get()
        logfile = self._log_var.get().strip() or None
        trajectory = self._traj_var.get().strip() or None

        # Get atoms from the GUI
        # (copy so we don't mutate the GUI's atoms mid-run)
        atoms = self.gui.atoms.copy()

        # Attach calculator
        try:
            calc = _make_calculator(calc_name)
        except Exception as exc:
            ui.error(_('Calculator error'), str(exc))
            return
        atoms.calc = calc

        self._max_steps = steps  # store so _handle_msg can compute progress %

        self._set_running(True)
        self._progress.config(mode='determinate', maximum=100, value=0)
        self._set_status_strip('#d0d0d0')
        self._status_var.set(_('Initializing…'))

        # Drain old queue entries
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Collect extra optimizer parameters from the dynamic panel
        extra_kwargs = {}
        for param_name, var in self._opt_params.items():
            raw = var.get().strip()
            try:
                # Try int first, then float
                extra_kwargs[param_name] = int(raw)
            except ValueError:
                try:
                    extra_kwargs[param_name] = float(raw)
                except ValueError:
                    ui.error(
                        _('Invalid input'),
                        f'Parameter {param_name!r} must be a number.',
                    )
                    return

        # Launch background thread
        self._thread = threading.Thread(
            target=self._worker,
            args=(atoms, opt_name, fmax, steps,
                  logfile, trajectory, extra_kwargs),
            daemon=True,
        )
        self._thread.start()

        # Start polling
        self._poll_queue()

    def _worker(self, atoms, opt_name, fmax, steps,
                logfile, trajectory, extra_kwargs):
        """Background thread: runs the optimization.

        Posts events to _queue for the main thread to read.
        """
        try:
            opt = _make_optimizer(
                opt_name, atoms, logfile, trajectory, **extra_kwargs
            )
        except Exception as exc:
            self._queue.put(('error', f'Optimizer error: {exc}'))
            return

        try:
            converged = False
            for converged in opt.irun(fmax=fmax, steps=steps):
                self._queue.put(('step', opt.nsteps, converged))

            # Final status
            if converged:
                self._queue.put(('done', atoms, True))
            else:
                self._queue.put(('done', atoms, False))

        except Exception as exc:
            self._queue.put(('error', str(exc)))

    def _poll_queue(self):
        """Called every 100 ms from the tkinter main loop to drain _queue."""
        try:
            while True:
                msg = self._queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass

        # Keep polling while thread is alive OR queue is not empty
        if self._thread is not None and self._thread.is_alive():
            self._poll_id = self.win.after(100, self._poll_queue)
        else:
            # One last drain after thread exits
            try:
                while True:
                    msg = self._queue.get_nowait()
                    self._handle_msg(msg)
            except queue.Empty:
                pass
            self._poll_id = None

    def _handle_msg(self, msg):
        kind = msg[0]

        if kind == 'step':
            _kind, nsteps, converged = msg
            pct = min(100, int(nsteps / max(1, self._max_steps) * 100))
            self._progress['value'] = pct
            self._status_var.set(
                f'Running… step {nsteps} / {self._max_steps}'
            )

        elif kind == 'done':
            _kind, relaxed_atoms, converged = msg
            self._progress.stop()
            self._progress.config(mode='determinate', value=100)
            self._set_running(False)

            if converged:
                self._progress['value'] = 100
                self._set_status_strip('#27ae60')   # green
                self._status_var.set('✓ Converged!')
            else:
                self._set_status_strip('#e67e22')   # orange
                self._status_var.set(
                    '⚠ Did not converge (max steps reached).'
                )

            # Push relaxed structure into the GUI
            self.gui.new_atoms(relaxed_atoms)
            self.gui.draw()

        elif kind == 'error':
            _kind, message = msg
            self._progress.stop()
            self._progress.config(mode='determinate', value=0)
            self._set_running(False)
            self._set_status_strip('#e74c3c')       # red
            self._status_var.set('✗ Error: ' + message)
            ui.error('Optimization error', message)

    # -----------------------------------------------------------------------
    # Window close
    # -----------------------------------------------------------------------

    def _on_close(self):
        # Cancel pending poll
        if self._poll_id is not None:
            self.win.after_cancel(self._poll_id)
        self.win.destroy()
