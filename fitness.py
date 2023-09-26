import multiprocessing
from multiprocessing import Process
import runn
import numpy as np

# Constants for energy unit conversion
Ha_to_eV = 27.211

# Reference energies for different species
E_At = -263.0812543
E_Atp = -262.7419035
E_At2p = -262.0950157
E_Atn = -263.1663307

# Reference energies for different configurations
E_At_m_calc = np.array([E_At2p, E_Atp, E_At, E_Atn])
E_At2 = np.array([-526.1867749, -526.191208, -526.1925101, -526.1916841, -526.1894469])
E_HAt = np.array([-263.6547155, -263.6666387, -263.670009, -263.6677744, -263.6619657])

# Fitness function that runs NWChem calculations for multiple atoms and molecules
def fitness(
    bas,
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
    # Create separate processes to run NWChem calculations for different species

    # Process for At
    p1 = Process(
        target=runn.run_many_calc_atom,
        args=(
            cal_dir,
            scratch_dir1,
            memory,
            bas,
            At_pp,
            At_m_calc_title,
            At_m_calc_charge,
            At_m_calc_mult,
            At_geometry,
            xcf,
            basenameAt,
        ),
    )

    # Process for At2
    p2 = Process(
        target=runn.run_many_calc_atom,
        args=(
            cal_dir,
            scratch_dir2,
            memory,
            bas,
            At_pp,
            At2_m_calc_title,
            At2_m_calc_charge,
            At2_m_calc_mult,
            At2_geometry,
            xcf,
            basenameAt2,
        ),
    )

    # Process for HAt
    p3 = Process(
        target=runn.run_many_calc_mol,
        args=(
            cal_dir,
            scratch_dir3,
            memory,
            bas,
            At_pp,
            hat_m_calc_title,
            hat_m_calc_charge,
            hat_m_calc_mult,
            HAt_geometry,
            xcf,
            basenameHAt,
            H_basis,
        ),
    )

    # Start the processes
    p1.start()
    p2.start()
    p3.start()

    # Wait for the processes to finish
    p1.join()
    p2.join()
    p3.join()

    # Retrieve the energies from the NWChem calculations
    E_out_at_m_calc, nwfail1 = runn.get_energy_many_calc(basenameAt, N_At)
    E_out_at2_m_calc, nwfail3 = runn.get_energy_many_calc(basenameAt2, N_At2)
    E_out_hat_m_calc, nwfail2 = runn.get_energy_many_calc(basenameHAt, N_HAt)

    # Convert the energies to NumPy arrays
    E_out_at_m_calc = np.array(E_out_at_m_calc)
    E_out_at2_m_calc = np.array(E_out_at2_m_calc)
    E_out_hat_m_calc = np.array(E_out_hat_m_calc)

    # Calculate the errors by comparing computed energies to reference energies
    errors = np.sum(abs(E_out_at_m_calc - E_At_m_calc))
    errors += np.sum(abs(E_out_at2_m_calc - E_At2))
    errors += np.sum(abs(E_out_hat_m_calc - E_HAt))
    errors *= Ha_to_eV / (N_At + N_At2 + N_HAt)

    return errors

# The provided code defines a fitness function that runs NWChem calculations
# for multiple atoms and molecules and computes the error based on reference energies.
