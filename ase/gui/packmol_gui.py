import os
import threading
import subprocess
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename

import ase.gui.ui as ui
from ase.io import read
import numpy as np


class PackmolWindow:
    def __init__(self, gui):
        self.gui = gui
        
        # 1. Setup the main popup window
        self.win = tk.Toplevel()
        self.win.title("Packmol Builder")
        self.win.geometry("400x300")
        
        # 2. Variables to store what the user types
        self.type_var = tk.StringVar(value="Bulk")
        self.density_var = tk.StringVar(value="1.0") # density in g/cm^3
        self.mol_path = tk.StringVar() # path to the molecule .xyz
        self.mol_count = tk.StringVar(value="500") # how many copies
        self.vacuum_var = tk.StringVar(value="15.0") # 15 Angstroms of vacuum above the slab
        
        self._build_ui()

    def _build_ui(self):
        """Draws the buttons and text boxes on the screen."""
        grid = tk.Frame(self.win, padx=10, pady=10)
        grid.pack(fill=tk.BOTH, expand=True)

        row = 0
        # --- Type Dropdown ---
        tk.Label(grid, text="Structure Type:").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Combobox(grid, textvariable=self.type_var, values=["Bulk", "Slab"], state="readonly").grid(row=row, column=1, sticky='ew')
        
        row += 1
        # --- Vacuum Height (only relevant for Slab) ---
        tk.Label(grid, text="Vacuum Height (Å):").grid(row=row, column=0, sticky='w', pady=5)
        tk.Entry(grid, textvariable=self.vacuum_var).grid(row=row, column=1, sticky='ew')

        row += 1
        # --- Density Input ---
        tk.Label(grid, text="Density (g/cm³):").grid(row=row, column=0, sticky='w', pady=5)
        tk.Entry(grid, textvariable=self.density_var).grid(row=row, column=1, sticky='ew')
        
        row += 1
        # --- Molecule File Picker ---
        tk.Label(grid, text="Molecule File:").grid(row=row, column=0, sticky='w', pady=5)
        tk.Entry(grid, textvariable=self.mol_path).grid(row=row, column=1, sticky='ew')
        tk.Button(grid, text="Browse", command=self._browse_file).grid(row=row, column=2, padx=5)

        row += 1
        # --- Molecule Count ---
        tk.Label(grid, text="Number of Molecules:").grid(row=row, column=0, sticky='w', pady=5)
        tk.Entry(grid, textvariable=self.mol_count).grid(row=row, column=1, sticky='ew')
        
        row += 1
        # --- Run Button ---
        tk.Button(self.win, text="Generate Structure", command=self._on_run, bg="#4CAF50", fg="white").pack(pady=10)
        
    def _browse_file(self):
        """Opens a file explorer to select a molecule structure file."""
        filepath = askopenfilename(title="Select Molecule", filetypes=[("XYZ Files", "*.xyz"), ("All Files", "*.*")])
        if filepath:
            self.mol_path.set(filepath)
            
    def _on_run(self):
        path = self.mol_path.get()
        #file path validation
        if not path:
            ui.error("Input Error", "Please select a molecule file.")
            return
        #molecule validation, density validation, and vacuum height validation
        try: 
            count = int(self.mol_count.get())
        except ValueError:
            ui.error("Input Error", "Number of molecules must be a whole number.")
            return
        try:
            density = float(self.density_var.get())
        except ValueError:
            ui.error("Input Error", "Density must be a number.")
            return
        try:
            vacuum = float(self.vacuum_var.get())
        except ValueError:
            ui.error("Input Error", "Vacuum height must be a number.")
            return
        structure_type = self.type_var.get()       # read from the dropdown

        # Range validation
        if not os.path.isfile(path):
            ui.error("Input Error", f"File not found: {path}")
            return
        if count < 1 or count > 100000:
            ui.error("Input Error", "Number of molecules must be between 1 and 100,000.")
            return
        if density <= 0:
            ui.error("Input Error", "Density must be a positive number.")
            return
        if vacuum < 0 or vacuum > 200:
            ui.error("Input Error", "Vacuum height must be between 0 and 200 Å.")
            return

        # 1. Read the molecule the user provided
        try:
            mol_atoms = read(path)
        except Exception as e:
            ui.error("Error", f"Failed to read molecule file: {e}")
            return

        # 2. Get the box size required
        Lx, Ly, Lz = self._calculate_box_dimension(
            mol_atoms, count, density, structure_type, vacuum)

        # 3. Run Packmol in a background thread so the GUI doesn't freeze!
        threading.Thread(
            target=self._run_packmol_thread,
            args=(path, count, Lx, Ly, Lz, structure_type),
            daemon=True,
        ).start()

    def _run_packmol_thread(self, mol_path, count, Lx, Ly, Lz, structure_type):
        import tempfile
        import shutil as _shutil

        tmpdir = tempfile.mkdtemp(prefix="ase_packmol_")
        try:
            # Packmol's Fortran parser truncates paths at spaces.
            # Copy the molecule file into a temp dir (guaranteed
            # space-free on Windows) and use simple filenames.
            mol_basename = os.path.basename(mol_path)
            tmp_mol = os.path.join(tmpdir, mol_basename)
            _shutil.copy2(mol_path, tmp_mol)

            output_name = "packmol_out.xyz"
            inp_content = (
                "tolerance 2.0\n"
                "filetype xyz\n"
                f"output {output_name}\n"
                "\n"
                f"structure {mol_basename}\n"
                f"  number {count}\n"
                f"  inside box 0. 0. 0. {Lx:.2f} {Ly:.2f} {Lz:.2f}\n"
                "end structure\n"
            )

            inp_path = os.path.join(tmpdir, "input.inp")
            with open(inp_path, "w") as f:
                f.write(inp_content)

            # Resolve the packmol executable.
            try:
                from packmol.cli import get_binary_path
                packmol_exe = str(get_binary_path())
            except (ImportError, FileNotFoundError):
                packmol_exe = (
                    os.environ.get("ASE_PACKMOL_COMMAND")
                    or _shutil.which("packmol")
                    or "packmol"
                )

            # On Windows the pip-built binary may need MinGW runtime
            # DLLs that sit next to the executable.
            env = os.environ.copy()
            bindir = os.path.dirname(packmol_exe)
            env["PATH"] = bindir + os.pathsep + env.get("PATH", "")

            with open(inp_path, "r") as stdin:
                subprocess.run(
                    [packmol_exe], stdin=stdin, check=True,
                    env=env, cwd=tmpdir,
                )

            # Read the result and push it into the GUI
            out_path = os.path.join(tmpdir, output_name)
            new_atoms = read(out_path)

            # Set the simulation cell (the bounding box)
            new_atoms.set_cell([Lx, Ly, Lz])

            # Set periodic boundary conditions
            if structure_type == "Bulk":
                new_atoms.pbc = [True, True, True]
            else:  # Slab
                new_atoms.pbc = [True, True, False]

            self.gui.new_atoms(new_atoms)
            self.gui.draw()
            print("Successfully generated and loaded the structure!")

        except Exception as e:
            print(f"Packmol failed: {e}. "
                  "Is packmol installed and in your PATH?")
        finally:
            _shutil.rmtree(tmpdir, ignore_errors=True)


    def _calculate_box_dimension(self, molecule_atoms, num_molecules, density,
                                 structure_type, vacuum):
        """
        Calculates box dimensions (Lx, Ly, Lz) in Angstroms for a target density.
        """
        # Mass of one molecule in atomic mass units (amu)
        mass_amu = sum(molecule_atoms.get_masses())
        total_mass_amu = mass_amu * num_molecules

        # 1 amu = 1.660539 x 10^-24 grams
        # 1 cm^3 = 10^24 Angstroms^3
        # Volume (A^3) = (total_mass_amu / density) * 1.660539
        volume_A3 = (total_mass_amu / density) * 1.660539

        if structure_type == "Bulk":
            side = volume_A3 ** (1/3)
            return side, side, side
        else:  # Slab
            # Make the slab Z-height ~1/3 of what a cube would be
            cube_side = volume_A3 ** (1/3)
            z_height = cube_side / 3.0
            xy_area = volume_A3 / z_height
            xy_side = xy_area ** 0.5
            return xy_side, xy_side, z_height + vacuum
