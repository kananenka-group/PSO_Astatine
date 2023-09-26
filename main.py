import os
import sys
import multiprocessing
from multiprocessing import Process
import numpy as np
import random
import copy
import basis
import runn
from particle import Particle
from fitness import fitness

import argparse
import json

# Define the main function that serves as the entry point of the program.
def main():
    # Define a conversion factor from Hartree (Ha) to electron volts (eV).
    Ha_to_eV = 27.211

    # Define energy values for different atomic states (in Ha).
    E_At = -263.0812543
    E_Atp = -262.7419035
    E_At2p = -262.0950157
    E_Atn = -263.1663307

    # Store energy values in an array for later reference.
    E_At_m_calc = np.array([E_At2p, E_Atp, E_At, E_Atn])

    # Define distances for various atomic configurations.
    R_At2 = [2.77842834, 2.87842834, 2.97842834, 3.07842834, 3.17842834]
    R_HAt = [1.5237857, 1.6237857, 1.7237857, 1.8237857, 1.9237857]

    # Store energy values corresponding to different distances in arrays.
    E_At2 = np.array([-526.1867749, -526.191208, -526.1925101, -526.1916841, -526.1894469])
    E_HAt = np.array([-263.6547155, -263.6666387, -263.670009, -263.6677744, -263.6619657])

    # Determine the number of atomic states and distances.
    N_At = len(E_At_m_calc)
    N_At2 = len(E_At2)
    N_HAt = len(E_HAt)

    # Define atomic properties for an element (At).
    At_EA = 2.412
    At_IP = 9.31751

    # Parse command-line arguments to obtain an input JSON file.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="input json file")
    args = parser.parse_args()

    # Read and load parameters from the input JSON file.
    with open(args.input_file) as json_file:
        j = json.load(json_file)

    # Extract various parameters from the JSON configuration.
    n_gen = j["generations"]
    n_ind = j["population_size"]
    save_n_best = j["save_n_best"]
    fittest_win = j["fittest_win_probability"] / 100.0
    job_type = j["job_type"]
    save_file = j["chk_file"]
    input_basis_file = j["input_basis_file"]
    H_basis_file = j["H_basis_file"]
    At_pp_file = j["At_pp_file"]
    mut_file = j["mutation_file"]
    mut_rule = j["mutation_impl"]
    cro_rule = j["crossover_impl"]

    # Extract mutation-related parameters from the JSON configuration.
    m_lik = j["mutation_probability"]  # Mutation probability (percentage)

    # Define the working directory.
    cal_dir = os.getcwd()

    # Create scratch directories for calculations.
    scratch_dir1 = "./tmp1"
    scratch_dir2 = "./tmp2"
    scratch_dir3 = "./tmp3"

    # Define memory allocation for calculations (in MB).
    memory = 32000

    # Define the exchange-correlation functional.
    xcf = "pbe0"

    # Define atomic titles for various states and configurations.
    At_title = "At"
    Atn_title = "At-"
    Atp_title = "At+"
    At2p_title = "At2+"  # Added title for At2+
    At_m_calc_title = [At2p_title, Atp_title, At_title, Atn_title]
    At2_m_calc_title = ["At2_%f" % (r) for r in R_At2]
    HAt_m_calc_title = ["HAt_%f" % (r) for r in R_HAt]

    # Define atomic charges for various states and configurations.
    At_charge = 0
    Atn_charge = -1
    Atp_charge = 1
    At2p_charge = 2
    At2_charge = 0
    HAt_charge = 0
    AtI_charge = 0
    At_m_calc_charge = [At2p_charge, Atp_charge, At_charge, Atn_charge]
    At2_m_calc_charge = [At2_charge for l in range(len(R_At2))]
    HAt_m_calc_charge = [HAt_charge for l in range(len(R_HAt))]

    # Define multiplicities for various atomic states and configurations.
    At_mult = 2
    Atn_mult = 1
    Atp_mult = 3
    At2p_mult = 4
    At2_mult = 1
    HAt_mult = 1
    At_m_calc_mult = [At2p_mult, Atp_mult, At_mult, Atn_mult]
    At2_m_calc_mult = [At2_mult for l in range(len(R_At2))]
    HAt_m_calc_mult = [HAt_mult for l in range(len(R_HAt))]

    # Define atomic geometries for various states and configurations.
    At_geometry = [
        "At   .10000000   0.0   0.0",
        "At   0.0   0.0   0.0",
        "At   .10000000   0.0   0.0",
        "At   0.0   0.0   0.0",
    ]
    At2_geometry = [
        "At  0.00000000     0.00000000     0.00000000\n At    0.00000000     0.00000000     %f"
        % (r)
        for r in R_At2
    ]
    HAt_geometry = [
        "H  0.00000000     0.00000000     0.00000000\n At    0.00000000     0.00000000     %f"
        % (r)
        for r in R_HAt
    ]

    # Define basenames for files.
    basenameAt = "nwcat"
    basenameAt2 = "nwcat2"
    basenameHAt = "nwchat"

    # Initialize variables for energy calculations.
    emin = 1e20
    gens = -1
    inds = -1
    eoriginal = 0.0

    # Create scratch directories if they do not exist.
    if not os.path.exists(scratch_dir1):
        os.mkdir(scratch_dir1)
    if not os.path.exists(scratch_dir2):
        os.mkdir(scratch_dir2)
    if not os.path.exists(scratch_dir3):
        os.mkdir(scratch_dir3)

    # Define the name of the output file.
    out_file_name = "ga.out"
    fo = open(out_file_name, "w")
import os
import sys
import multiprocessing
from multiprocessing import Process
import numpy as np
import random
import copy
import basis
import runn
from particle import Particle
from fitness import fitness

import argparse
import json

# Define the main function that serves as the entry point of the program.
def main():
    # Define a conversion factor from Hartree (Ha) to electron volts (eV).
    Ha_to_eV = 27.211

    # Define energy values for different atomic states (in Ha).
    E_At = -263.0812543
    E_Atp = -262.7419035
    E_At2p = -262.0950157
    E_Atn = -263.1663307

    # Store energy values in an array for later reference.
    E_At_m_calc = np.array([E_At2p, E_Atp, E_At, E_Atn])

    # Define distances for various atomic configurations.
    R_At2 = [2.77842834, 2.87842834, 2.97842834, 3.07842834, 3.17842834]
    R_HAt = [1.5237857, 1.6237857, 1.7237857, 1.8237857, 1.9237857]

    # Store energy values corresponding to different distances in arrays.
    E_At2 = np.array([-526.1867749, -526.191208, -526.1925101, -526.1916841, -526.1894469])
    E_HAt = np.array([-263.6547155, -263.6666387, -263.670009, -263.6677744, -263.6619657])

    # Determine the number of atomic states and distances.
    N_At = len(E_At_m_calc)
    N_At2 = len(E_At2)
    N_HAt = len(E_HAt)

    # Define atomic properties for an element (At).
    At_EA = 2.412
    At_IP = 9.31751

    # Parse command-line arguments to obtain an input JSON file.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="input json file")
    args = parser.parse_args()

    # Read and load parameters from the input JSON file.
    with open(args.input_file) as json_file:
        j = json.load(json_file)

    # Extract various parameters from the JSON configuration.
    n_gen = j["generations"]
    n_ind = j["population_size"]
    save_n_best = j["save_n_best"]
    fittest_win = j["fittest_win_probability"] / 100.0
    job_type = j["job_type"]
    save_file = j["chk_file"]
    input_basis_file = j["input_basis_file"]
    H_basis_file = j["H_basis_file"]
    At_pp_file = j["At_pp_file"]
    mut_file = j["mutation_file"]
    mut_rule = j["mutation_impl"]
    cro_rule = j["crossover_impl"]

    # Extract mutation-related parameters from the JSON configuration.
    m_lik = j["mutation_probability"]  # Mutation probability (percentage)

    # Define the working directory.
    cal_dir = os.getcwd()

    # Create scratch directories for calculations.
    scratch_dir1 = "./tmp1"
    scratch_dir2 = "./tmp2"
    scratch_dir3 = "./tmp3"

    # Define memory allocation for calculations (in MB).
    memory = 32000

    # Define the exchange-correlation functional.
    xcf = "pbe0"

    # Define atomic titles for various states and configurations.
    At_title = "At"
    Atn_title = "At-"
    Atp_title = "At+"
    At2p_title = "At2+"  # Added title for At2+
    At_m_calc_title = [At2p_title, Atp_title, At_title, Atn_title]
    At2_m_calc_title = ["At2_%f" % (r) for r in R_At2]
    HAt_m_calc_title = ["HAt_%f" % (r) for r in R_HAt]

    # Define atomic charges for various states and configurations.
    At_charge = 0
    Atn_charge = -1
    Atp_charge = 1
    At2p_charge = 2
    At2_charge = 0
    HAt_charge = 0
    AtI_charge = 0
    At_m_calc_charge = [At2p_charge, Atp_charge, At_charge, Atn_charge]
    At2_m_calc_charge = [At2_charge for l in range(len(R_At2))]
    HAt_m_calc_charge = [HAt_charge for l in range(len(R_HAt))]

    # Define multiplicities for various atomic states and configurations.
    At_mult = 2
    Atn_mult = 1
    Atp_mult = 3
    At2p_mult = 4
    At2_mult = 1
    HAt_mult = 1
    At_m_calc_mult = [At2p_mult, Atp_mult, At_mult, Atn_mult]
    At2_m_calc_mult = [At2_mult for l in range(len(R_At2))]
    HAt_m_calc_mult = [HAt_mult for l in range(len(R_HAt))]

    # Define atomic geometries for various states and configurations.
    At_geometry = [
        "At   .10000000   0.0   0.0",
        "At   0.0   0.0   0.0",
        "At   .10000000   0.0   0.0",
        "At   0.0   0.0   0.0",
    ]
    At2_geometry = [
        "At  0.00000000     0.00000000     0.00000000\n At    0.00000000     0.00000000     %f"
        % (r)
        for r in R_At2
    ]
    HAt_geometry = [
        "H  0.00000000     0.00000000     0.00000000\n At    0.00000000     0.00000000     %f"
        % (r)
        for r in R_HAt
    ]

    # Define basenames for files.
    basenameAt = "nwcat"
    basenameAt2 = "nwcat2"
    basenameHAt = "nwchat"

    # Initialize variables for energy calculations.
    emin = 1e20
    gens = -1
    inds = -1
    eoriginal = 0.0

    # Create scratch directories if they do not exist.
    if not os.path.exists(scratch_dir1):
        os.mkdir(scratch_dir1)
    if not os.path.exists(scratch_dir2):
        os.mkdir(scratch_dir2)
    if not os.path.exists(scratch_dir3):
        os.mkdir(scratch_dir3)

    # Define the name of the output file.
    out_file_name = "ga.out"
    fo = open(out_file_name, "w")

    # Check if this is a new calculation or a continuation.
    if job_type == 0:
        fo.write(" Starting new calculation \n")
        bas_start = basis.read_start_basis(
            input_basis_file
        )  # Load the initial basis set from the input file

        fo.write(" -------- Input basis -------- \n")
        # Print the initial basis set to the output file.
        for p in range(len(bas_start)):
            v = bas_start[p]
            fo.write("   %s     %18.9f \n" % (v[0], v[1]))
        fo.write(" ------ Input basis end ------ \n\n")

    # Read and load the pseudopotential.
    fo.write(" Reading pseudopotential from %s \n" % (At_pp_file))
    At_pp = basis.read_pseudopotential(At_pp_file)

    fo.write(" -------- Pseudopotential ----------\n")
    # Print the pseudopotential to the output file.
    fo.write(At_pp)
    fo.write(" -------- Pseudopotential end ----------\n\n")

    # Read H and I basis sets and pseudopotential.
    H_basis = basis.read_atom_basis(H_basis_file)

    # Clone the input basis into n_ind.
    original_basis_mut = False
    sigma0 = 0

    # Particle swarm optimization parameters.
    w = 0.829  # Inertia
    c1 = 1.49445  # Cognitive (particle)
    c2 = 1.49445  # Social (swarm)
    n = 15  # Number of particles
    max_iter = 200  # Maximum number of iterations
    rnd = random.Random(0)

    # Create n random particles for the particle swarm.
    swarm = [
        Particle(
            fitness,
            bas_start,
            i,
            cal_dir,
            scratch_dir1,
            memory,
            At_pp,
            At_m_calc_title,
            At_m_calc_charge,
            At_m_calc_mult,
            At_geometry,
            xcf,
            basenameAt,
            scratch_dir2,
            At2_m_calc_title,
            At2_m_calc_charge,
            At2_m_calc_mult,
            At2_geometry,
            basenameAt2,
            scratch_dir3,
            HAt_m_calc_title,
            HAt_m_calc_charge,
            HAt_m_calc_mult,
            HAt_geometry,
            basenameHAt,
            H_basis,
            N_At,
            N_At2,
            N_HAt,
        )
        for i in range(n)
    ]

    # Initialize variables for the best swarm position and fitness.
    dim = len(bas_start)
    best_swarm_pos = [0.0 for i in range(dim)]
    best_swarm_fitnessVal = sys.float_info.max  # Best fitness value in the swarm
    best_swarm_bas = [["", 0.0] for i in range(dim)]

    # Find the best particle in the swarm and its fitness.
    for i in range(n):  # Check each particle
        if swarm[i].fitness < best_swarm_fitnessVal:
            cp_best_swarm_pos = copy.deepcopy(swarm[i])
            best_swarm_fitnessVal = cp_best_swarm_pos.fitness
            best_swarm_pos = cp_best_swarm_pos.position
            best_swarm_bas = cp_best_swarm_pos.bas

    # Main loop of particle swarm optimization.
    Iter = 0
    while Iter < max_iter:
        fo.write(
            "\n\n <<<<<<<<<<<<<<<< Beginning of a new iteration %d out of %d >>>>>>>>>>>>>>>>>>>\n\n"
            % ((Iter + 1), max_iter)
        )
        fo.flush()
        fo.write("Iter = " + str(Iter) + " best fitness = %.7f" % best_swarm_fitnessVal)
        basis.write_basis(best_swarm_bas, Iter, 0)

        for i in range(n):  # Process each particle
            # Compute the new velocity of the current particle.
            for k in range(dim):
                r1 = rnd.random()  # Randomizations
                r2 = rnd.random()

                swarm[i].velocity[k] = (
                    (w * swarm[i].velocity[k])
                    + (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k]))
                    + (c2 * r2 * (best_swarm_pos[k] - swarm[i].position[k]))
                )

            # Compute the new position using the new velocity.
            new_position = []
            boundaries_low = [14, 29, 42, 48]
            boundaries_up = [0, 15, 30, 43]
            for k in range(dim):
                new_position.append(swarm[i].position[k] + swarm[i].velocity[k])
                if (k not in boundaries_low) and (k not in boundaries_up):
                    if new_position[-1] > swarm[i].position[k - 1] / 1.1:
                        new_position[-1] = (
                            2 * swarm[i].position[k - 1] / 1.1 - new_position[-1]
                        )
                    elif new_position[-1] < swarm[i].position[k + 1] * 1.1:
                        new_position[-1] = (
                            2 * swarm[i].position[k + 1] * 1.1 - new_position[-1]
                        )
                    elif new_position[-1] > new_position[-2] / 1.1:
                        new_position[-1] = 2 * new_position[-2] / 1.1 - new_position[-1]
                    elif new_position[-1] <= 0.0:
                        new_position[-1] = -new_position[-1]
                elif k in boundaries_up:
                    if new_position[-1] < swarm[i].position[k + 1] * 1.1:
                        new_position[-1] = (
                            2 * swarm[i].position[k + 1] * 1.1 - new_position[-1]
                        )
                    if new_position[-1] > bas_start[k][1] * 2.0:
                        new_position[-1] = 2 * bas_start[k][1] * 2.0 - new_position[-1]
                    elif new_position[-1] <= 0.0:
                        new_position[-1] = -new_position[-1]
                elif k in boundaries_low:
                    if new_position[-1] > swarm[i].position[k - 1] / 1.1:
                        new_position[-1] = (
                            2 * swarm[i].position[k - 1] / 1.1 - new_position[-1]
                        )
                    elif new_position[-1] > new_position[-2] / 1.1:
                        new_position[-1] = 2 * new_position[-2] / 1.1 - new_position[-1]
                    elif new_position[-1] <= 0.0:
                        new_position[-1] = -new_position[-1]

            # Update the position of the current particle.
            swarm[i].update_position(new_position)
            swarm[i].update_bas()  # Update the basis set using the new parameters

            # Compute the fitness of the new position.
            swarm[i].fitness = fitness(
                swarm[i].bas,
                cal_dir,
                scratch_dir1,
                memory,
                At_pp,
                At_m_calc_title,
                At_m_calc_charge,
                At_m_calc_mult,
                At_geometry,
                xcf,
                basenameAt,
                scratch_dir2,
                At2_m_calc_title,
                At2_m_calc_charge,
                At2_m_calc_mult,
                At2_geometry,
                basenameAt2,
                scratch_dir3,
                HAt_m_calc_title,
                HAt_m_calc_charge,
                HAt_m_calc_mult,
                HAt_geometry,
                basenameHAt,
                H_basis,
                N_At,
                N_At2,
                N_HAt,
            )
            print(swarm[i].fitness)

            # Check if the new position is the best for the particle.
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = copy.copy(swarm[i].fitness)
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            # Check if the new position is the best overall.
            if swarm[i].fitness < best_swarm_fitnessVal:
                cp_best_swarm_pos = copy.deepcopy(swarm[i])
                best_swarm_fitnessVal = cp_best_swarm_pos.fitness
                best_swarm_pos = cp_best_swarm_pos.position
                best_swarm_bas = cp_best_swarm_pos.bas

        Iter += 1

    fo.close()

# Entry point of the program; calls the main function.
if __name__ == "__main__":
    main()

    # Check if this is a new calculation or a continuation.
    if job_type == 0:
        fo.write(" Starting new calculation \n")
        bas_start = basis.read_start_basis(
            input_basis_file
        )  # Load the initial basis set from the input file

        fo.write(" -------- Input basis -------- \n")
        # Print the initial basis set to the output file.
        for p in range(len(bas_start)):
            v = bas_start[p]
            fo.write("   %s     %18.9f \n" % (v[0], v[1]))
        fo.write(" ------ Input basis end ------ \n\n")

    # Read and load the pseudopotential.
    fo.write(" Reading pseudopotential from %s \n" % (At_pp_file))
    At_pp = basis.read_pseudopotential(At_pp_file)

    fo.write(" -------- Pseudopotential ----------\n")
    # Print the pseudopotential to the output file.
    fo.write(At_pp)
    fo.write(" -------- Pseudopotential end ----------\n\n")

    # Read H and I basis sets and pseudopotential.
    H_basis = basis.read_atom_basis(H_basis_file)

    # Clone the input basis into n_ind.
    original_basis_mut = False
    sigma0 = 0

    # Particle swarm optimization parameters.
    w = 0.829  # Inertia
    c1 = 1.49445  # Cognitive (particle)
    c2 = 1.49445  # Social (swarm)
    n = 15  # Number of particles
    max_iter = 200  # Maximum number of iterations
    rnd = random.Random(0)

    # Create n random particles for the particle swarm.
    swarm = [
        Particle(
            fitness,
            bas_start,
            i,
            cal_dir,
            scratch_dir1,
            memory,
            At_pp,
            At_m_calc_title,
            At_m_calc_charge,
            At_m_calc_mult,
            At_geometry,
            xcf,
            basenameAt,
            scratch_dir2,
            At2_m_calc_title,
            At2_m_calc_charge,
            At2_m_calc_mult,
            At2_geometry,
            basenameAt2,
            scratch_dir3,
            HAt_m_calc_title,
            HAt_m_calc_charge,
            HAt_m_calc_mult,
            HAt_geometry,
            basenameHAt,
            H_basis,
            N_At,
            N_At2,
            N_HAt,
        )
        for i in range(n)
    ]

    # Initialize variables for the best swarm position and fitness.
    dim = len(bas_start)
    best_swarm_pos = [0.0 for i in range(dim)]
    best_swarm_fitnessVal = sys.float_info.max  # Best fitness value in the swarm
    best_swarm_bas = [["", 0.0] for i in range(dim)]

    # Find the best particle in the swarm and its fitness.
    for i in range(n):  # Check each particle
        if swarm[i].fitness < best_swarm_fitnessVal:
            cp_best_swarm_pos = copy.deepcopy(swarm[i])
            best_swarm_fitnessVal = cp_best_swarm_pos.fitness
            best_swarm_pos = cp_best_swarm_pos.position
            best_swarm_bas = cp_best_swarm_pos.bas

    # Main loop of particle swarm optimization.
    Iter = 0
    while Iter < max_iter:
        fo.write(
            "\n\n <<<<<<<<<<<<<<<< Beginning of a new iteration %d out of %d >>>>>>>>>>>>>>>>>>>\n\n"
            % ((Iter + 1), max_iter)
        )
        fo.flush()
        fo.write("Iter = " + str(Iter) + " best fitness = %.7f" % best_swarm_fitnessVal)
        basis.write_basis(best_swarm_bas, Iter, 0)

        for i in range(n):  # Process each particle
            # Compute the new velocity of the current particle.
            for k in range(dim):
                r1 = rnd.random()  # Randomizations
                r2 = rnd.random()

                swarm[i].velocity[k] = (
                    (w * swarm[i].velocity[k])
                    + (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k]))
                    + (c2 * r2 * (best_swarm_pos[k] - swarm[i].position[k]))
                )

            # Compute the new position using the new velocity.
            new_position = []
            boundaries_low = [14, 29, 42, 48]
            boundaries_up = [0, 15, 30, 43]
            for k in range(dim):
                new_position.append(swarm[i].position[k] + swarm[i].velocity[k])
                if (k not in boundaries_low) and (k not in boundaries_up):
                    if new_position[-1] > swarm[i].position[k - 1] / 1.1:
                        new_position[-1] = (
                            2 * swarm[i].position[k - 1] / 1.1 - new_position[-1]
                        )
                    elif new_position[-1] < swarm[i].position[k + 1] * 1.1:
                        new_position[-1] = (
                            2 * swarm[i].position[k + 1] * 1.1 - new_position[-1]
                        )
                    elif new_position[-1] > new_position[-2] / 1.1:
                        new_position[-1] = 2 * new_position[-2] / 1.1 - new_position[-1]
                    elif new_position[-1] <= 0.0:
                        new_position[-1] = -new_position[-1]
                elif k in boundaries_up:
                    if new_position[-1] < swarm[i].position[k + 1] * 1.1:
                        new_position[-1] = (
                            2 * swarm[i].position[k + 1] * 1.1 - new_position[-1]
                        )
                    if new_position[-1] > bas_start[k][1] * 2.0:
                        new_position[-1] = 2 * bas_start[k][1] * 2.0 - new_position[-1]
                    elif new_position[-1] <= 0.0:
                        new_position[-1] = -new_position[-1]
                elif k in boundaries_low:
                    if new_position[-1] > swarm[i].position[k - 1] / 1.1:
                        new_position[-1] = (
                            2 * swarm[i].position[k - 1] / 1.1 - new_position[-1]
                        )
                    elif new_position[-1] > new_position[-2] / 1.1:
                        new_position[-1] = 2 * new_position[-2] / 1.1 - new_position[-1]
                    elif new_position[-1] <= 0.0:
                        new_position[-1] = -new_position[-1]

            # Update the position of the current particle.
            swarm[i].update_position(new_position)
            swarm[i].update_bas()  # Update the basis set using the new parameters

            # Compute the fitness of the new position.
            swarm[i].fitness = fitness(
                swarm[i].bas,
                cal_dir,
                scratch_dir1,
                memory,
                At_pp,
                At_m_calc_title,
                At_m_calc_charge,
                At_m_calc_mult,
                At_geometry,
                xcf,
                basenameAt,
                scratch_dir2,
                At2_m_calc_title,
                At2_m_calc_charge,
                At2_m_calc_mult,
                At2_geometry,
                basenameAt2,
                scratch_dir3,
                HAt_m_calc_title,
                HAt_m_calc_charge,
                HAt_m_calc_mult,
                HAt_geometry,
                basenameHAt,
                H_basis,
                N_At,
                N_At2,
                N_HAt,
            )
            print(swarm[i].fitness)

            # Check if the new position is the best for the particle.
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = copy.copy(swarm[i].fitness)
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            # Check if the new position is the best overall.
            if swarm[i].fitness < best_swarm_fitnessVal:
                cp_best_swarm_pos = copy.deepcopy(swarm[i])
                best_swarm_fitnessVal = cp_best_swarm_pos.fitness
                best_swarm_pos = cp_best_swarm_pos.position
                best_swarm_bas = cp_best_swarm_pos.bas

        Iter += 1

    fo.close()

# Entry point of the program; calls the main function.
if __name__ == "__main__":
    main()

