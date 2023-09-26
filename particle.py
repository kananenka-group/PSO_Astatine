import copy
import numpy as np
import random

# Define a Particle class to represent particles in the Particle Swarm Optimization (PSO) algorithm.
class Particle:
    # Initialize the particle object with various parameters.
    def __init__(
        self,
        fitness,
        bas0,
        seed,
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
        hat_m_calc_title,
        hat_m_calc_charge,
        hat_m_calc_mult,
        HAt_geometry,
        basenameHAt,
        H_basis,
        N_At,
        N_At2,
        N_HAt,
    ):
        self.rnd = random.Random(seed)  # Create a random number generator with the given seed.
        dim = len(bas0)  # Get the dimension of the basis set.
        self.bas = bas0  # Initialize the particle's basis set.
        
        # Initialize the position of the particle with small random perturbations.
        self.position = [
            bas0[i][1] * (1 + np.random.normal(0, 0.01)) for i in range(dim)
        ]

        # Initialize the velocity of the particle with small random values.
        self.velocity = [bas0[i][1] * np.random.normal(0, 0.02) for i in range(dim)]

        # Update the basis set based on the initial position.
        for i in range(len(self.position)):
            self.bas[i][1] = self.position[i]

        # Initialize the best particle position with the current position.
        self.best_part_pos = [0.0 for i in range(dim)]

        # Compute the fitness of the particle using the provided fitness function.
        self.fitness = fitness(
            self.bas,
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
            hat_m_calc_title,
            hat_m_calc_charge,
            hat_m_calc_mult,
            HAt_geometry,
            basenameHAt,
            H_basis,
            N_At,
            N_At2,
            N_HAt,
        )  # Current fitness value

        # Initialize the best position and fitness of this particle.
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness  # Best fitness value for this particle

    # Update the basis set of the particle.
    def update_bas(self):
        for i in range(len(self.position)):
            self.bas[i][1] = self.position[i]

    # Update the position of the particle.
    def update_position(self, new_position):
        for i in range(len(self.position)):
            self.position[i] = new_position[i]

