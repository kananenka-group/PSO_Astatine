# Program Execution Readme
This program optimized basis set exponents using Particle Swarm Optimization algorithm.
Author: Kennet Rueda Espinosa (kruedae@udel.edu), University of Delaware

## Software:
You will need NWChem and Python to run this code.

## Step 1: Bash Execution

Execute the PSO program with:
python main.py input.json

## Step 2: Choosing Initial Basis

You have the option to replace the file `Astatine_initial_basis.dat` with any of the files from the `initial_basis` folder. The `initial_basis` folder contains the initial basis sets of each size. You can choose the appropriate basis set for your calculations by replacing `Astatine_initial_basis.dat` in the program directory with your desired basis set file.

## Step 3: Customizing Input Parameters

You can also customize various input parameters by modifying the `input.json` file. In this file, you can adjust the core potentials and specify other basis sets. Make sure to follow the correct JSON syntax when making changes to the input file.

## Step 4: Identifying Target Energies

The target energies for the program are defined in the `main.py` file and the `fitness.py` file. If you need to adjust or access these target energies, please refer to these Python scripts for the relevant information.

If you have any questions or encounter any issues while using this program, please feel free to reach out 
Kennet Rueda Espinosa (kruedae@udel.edu) or Alexei Kananenka (akanane@udel.edu) for assistance.

Happy computing!
