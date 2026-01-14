import sys
from collections import Counter

import MDAnalysis as mda
import numpy as np
import scipy
from MDAnalysis import transformations
from rdkit import Chem
from rdkit.Chem import AllChem

np.set_printoptions(threshold=sys.maxsize)

# Professor Qi's Lab, Dartmouth College
# https://www.qimodeling.com/research


######################## change these parameters ###############################
# parameters to find bonds
# same algorithm that VMD uses: http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.1/ug/node26.html
# d_at1_at2 < 0.6*(R_at_1 + R_at_2)
fudge_factor_parm = 0.6

# loading tarjectory
filename_gro = "PE_Pt_centered.gro"
filename_xtc = "PE_Pt_centered_data.xtc"

output_file_path = "finding_fragments.out"

gro = mda.Universe(filename_gro, filename_xtc)

# PE number of carbons and chains
n_carbons = 50
n_chains = 40

type_1_mass = 195.084
type_2_mass = 12.011
type_3_mass = 1.008

# vdwradii for atoms, must be a dict of format {type:radii}
vdwradii_dic = {"Pt": 1.75, "C": 1.7, "H": 1.1}
# 1 195.084  # Pt
# 2 12.011  # C
# 3 1.008  # H
######################## change these parameters ###############################

# Open the output file
with open(output_file_path, "w") as output_file:
    # updating the masses according to the atom types
    Pt_atoms = gro.select_atoms("name Pt")
    Pt_atoms.types = "Pt"
    Pt_atoms.masses = type_1_mass

    C_atoms = gro.select_atoms("name C")
    C_atoms.masses = type_2_mass

    H_atoms = gro.select_atoms("name H")
    H_atoms.masses = type_3_mass

    # wrapping to the unit cell
    ag = gro.atoms
    transform = mda.transformations.wrap(ag)
    gro.trajectory.add_transformations(transform)

    # reference atoms
    polymer_atoms = gro.select_atoms("name C or name H")

    # print('#Simulation Box ([Lx, Ly, Lz, alpha, beta, gamma]):', gro.dimensions)
    # print("Time Frame and found compounds")

    for ts in mda.lib.log.ProgressBar(
        range(len(gro.trajectory)), verbose=True, total=len(gro.trajectory)
    ):
        # pointing to frame "ts"
        gro.trajectory[ts]

        time_line = (
            f"#Time = {gro.trajectory[ts].time:.3f} ps, found compounds (SMILES)"
        )
        # print(time_line)
        output_file.write(time_line + "\n")

        # finding bonds and accounting for periodic boundary conditions
        # same algorithm that VMD uses: http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.1/ug/node26.html
        polymer_bonds = mda.topology.guessers.guess_bonds(
            polymer_atoms,
            polymer_atoms.positions,
            box=gro.dimensions,
            vdwradii=vdwradii_dic,
            fudge_factor=fudge_factor_parm,
        )

        # loading a copy universe to find fragments,
        # this will be reloaded for each loop
        gro_copy = mda.Universe(filename_gro, filename_xtc)

        # pointing the copied universe to frame "ts"
        gro_copy.trajectory[ts]

        # adding bonds to the copied universe
        # this will be updated for each loop
        gro_copy.add_TopologyAttr("bonds", polymer_bonds)

        # selecting carbon and hydrogen atoms in the copied universe
        polymer_atoms_copy = gro_copy.select_atoms("name C or name H")

        # list to save SMILES string
        smiles_list = []

        for ij in polymer_atoms_copy.fragments:
            # convert fragments with more than 1 atom
            if len(ij) > 1:
                output_file_tmp = "tmp_structure.pdb"
                ij.atoms.write(output_file_tmp)

                mol = Chem.MolFromPDBFile(output_file_tmp, removeHs=False)

                # ensure the molecule is valid and generate the SMILES

                if mol:
                    # generate the SMILES string
                    smiles = Chem.MolToSmiles(mol)
                    # print("SMILES:", smiles)
                    smiles_list.append(smiles)

                else:
                    # print("#Failed to parse the structure with RDKit.")
                    smiles_list.append("[X]")

            else:
                tmp_at_name = "[" + str(ij.names[0]) + "]"
                smiles_list.append(tmp_at_name)

        # count unique SMILES and their frequencies
        smiles_counter = Counter(smiles_list)

        # printing SMILEs and their frequencies
        for smiles, freq in smiles_counter.items():
            result_line = f"{smiles}, {freq}"
            # print(result_line)
            output_file.write(result_line + "\n")

        ###sanity check
        # number of carbons in PE
        n_C_atoms_system = n_carbons * n_chains
        # number of H's in PE, 2H's per carbon and 2H's in the terminal carbons for
        # each chain
        n_H_atoms_system = n_carbons * n_chains * 2 + 2 * n_chains

        # counting the total number of carbon (C) and hydrogen (H) atoms
        # initialize counters
        total_C = 0
        total_H = 0

        # counting occurrences of "C" and "H" in each string
        for s in smiles_list:
            count = Counter(s)
            total_C += count["C"]
            total_H += count["H"]

        if n_C_atoms_system != total_C:
            error_message = (
                "###############################################################"
            )
            # print(error_message)
            output_file.write(error_message + "\n")

            error_message = f"#Total Number of found C atoms = {total_C:d}"
            # print(error_message)
            output_file.write(error_message + "\n")

            error_message = f"#Total Number of expected C atoms = {n_C_atoms_system:d}"
            # print(error_message)
            output_file.write(error_message + "\n")

            error_message = (
                "###############################################################"
            )
            # print(error_message)
            output_file.write(error_message + "\n")

        if n_H_atoms_system != total_H:
            error_message = (
                "###############################################################"
            )
            # print(error_message)
            output_file.write(error_message + "\n")

            error_message = f"#Total Number of found H atoms = {total_H:d}"
            # print(error_message)
            output_file.write(error_message + "\n")

            error_message = f"#Total Number of expected H atoms = {n_H_atoms_system:d}"
            # print(error_message)
            output_file.write(error_message + "\n")

            error_message = (
                "###############################################################"
            )
            # print(error_message)
            output_file.write(error_message + "\n")
        ###sanity check
