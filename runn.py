from subprocess import Popen, PIPE
import os
import sys

# Function to write an NWChem input file for a single calculation.
def write_nw_input_file(
    calc_dir,
    temp_dir,
    memory,
    basis,
    pp,
    title,
    charge,
    mult,
    geometry,
    xcf,
    basename,
    extra_basis=None,
):
    # Create a new NWChem input file with the given parameters.
    file_name = basename + ".nw"

    f = open(file_name, "w")
    f.write("PERMANENT_DIR %s \n" % (calc_dir))
    f.write("SCRATCH_DIR %s \n" % (temp_dir))
    f.write("MEMORY %d mb \n" % (memory))

    f.write('\nBASIS "ao basis" spherical\n')

    if extra_basis is not None:
        f.write(extra_basis)

    for bas in basis:
        sh = bas[0]
        ex = bas[1]
        f.write(" %s   %s  \n " % ("At", sh))
        f.write(" %15.7f  1.000   \n" % (ex))
    f.write("end\n")

    f.write("\n")
    f.write(pp)

    f.write("\n")
    f.write('title  "%s" \n' % (title))
    f.write("charge %7.5f \n" % (charge))

    f.write("\n")
    f.write("geometry units angstrom\n")
    f.write("symmetry group c1\n")
    f.write(" %s \n" % (geometry))
    f.write("end\n")

    f.write("\n")
    f.write("dft \n")
    f.write("grid xfine\n")
    f.write("mult %d \n" % (mult))
    f.write("odft\n")
    f.write("iterations 100\n")
    f.write("convergence energy 1.0E-9\n")  # Convergence criteria for energy
    f.write("convergence density 5.00000E-07\n")  # Convergence criteria for density
    f.write("convergence gradient 5.00000E-07\n")  # Convergence criteria for gradient
    f.write("xc %s  \n" % (xcf))
    f.write("end \n")
    f.write("task sodft ")

    f.close()

# Function to run an NWChem calculation for a single calculation.
def run_nwchem(basename):
    s = "mpirun -np 12 nwchem " + basename + ".nw > " + basename + ".out"
    # s = "nwchem " + basename + ".nw > " + basename + ".out"
    os.system(s)
    s = "rm " + basename + ".{movecs,db,evals}"
    os.system(s)

# Function to retrieve the energy from the output file of a single calculation.
def get_energy(basename):
    s = "grep 'Total SO-DFT' " + basename + ".out"
    process = Popen(s, shell=True, stdout=PIPE)
    stdout, stderr = process.communicate()

    nwfail = False
    err = process.returncode
    if err == 0:
        line = stdout.split()
        energy = float(line[4])
    else:
        print("NWChem failed %s" % (str(err)))
        energy = -1e10
        nwfail = True

    return energy, nwfail

# Function to run an NWChem calculation for a single atom.
def run_atom(
    calc_dir, temp_dir, memory, basis, pp, title, charge, mult, geometry, xcf, basename
):
    write_nw_input_file(
        calc_dir,
        temp_dir,
        memory,
        basis,
        pp,
        title,
        charge,
        mult,
        geometry,
        xcf,
        basename,
    )
    run_nwchem(basename)

# Function to run multiple calculations for a single atom.
def run_many_calc_atom(
    calc_dir, temp_dir, memory, basis, pp, title, charge, mult, geometry, xcf, basename
):
    write_many_calc_nw_input_file(
        calc_dir,
        temp_dir,
        memory,
        basis,
        pp,
        title,
        charge,
        mult,
        geometry,
        xcf,
        basename,
    )
    run_nwchem(basename)

# Function to run an NWChem calculation for a molecule.
def run_mol(
    calc_dir,
    temp_dir,
    memory,
    basis,
    basis2,
    pp,
    title,
    charge,
    mult,
    geometry,
    xcf,
    basename,
):
    write_nw_input_file(
        calc_dir,
        temp_dir,
        memory,
        basis,
        pp,
        title,
        charge,
        mult,
        geometry,
        xcf,
        basename,
        extra_basis=basis2,
    )
    run_nwchem(basename)

# Function to run multiple calculations for a molecule.
def run_many_calc_mol(
    calc_dir,
    temp_dir,
    memory,
    basis,
    pp,
    title,
    charge,
    mult,
    geometry,
    xcf,
    basename,
    basis2,
):
    write_many_calc_nw_input_file(
        calc_dir,
        temp_dir,
        memory,
        basis,
        pp,
        title,
        charge,
        mult,
        geometry,
        xcf,
        basename,
        extra_basis=basis2,
    )
    run_nwchem(basename)

# Function to write NWChem input files for multiple calculations.
def write_many_calc_nw_input_file(
    calc_dir,
    temp_dir,
    memory,
    basis,
    pp,
    title,
    charge,
    mult,
    geometry,
    xcf,
    basename,
    extra_basis=None,
):
    file_name = basename + ".nw"

    f = open(file_name, "w")
    f.write("PERMANENT_DIR %s \n" % (calc_dir))
    f.write("SCRATCH_DIR %s \n" % (temp_dir))
    f.write("MEMORY %d mb \n" % (memory))

    f.write('\nBASIS "ao basis" spherical\n')

    if extra_basis is not None:
        f.write(extra_basis)

    for bas in basis:
        sh = bas[0]
        ex = bas[1]
        f.write(" %s   %s  \n " % ("At", sh))
        f.write(" %15.7f  1.000   \n" % (ex))
    f.write("end\n")

    f.write("\n")
    f.write(pp)
    for i in range(len(title)):
        f.write("\n")
        f.write('title  "%s" \n' % (title[i]))
        f.write("charge %7.5f \n" % (charge[i]))

        f.write("\n")
        f.write("geometry units angstrom\n")
        f.write("symmetry group c1\n")
        f.write(" %s \n" % (geometry[i]))
        f.write("end\n")

        f.write("\n")
        f.write("dft \n")
        f.write("grid xfine\n")
        f.write("mult %d \n" % (mult[i]))
        f.write("odft\n")
        f.write("iterations 100\n")
        f.write("convergence energy 1.0E-9\n")
        f.write("convergence density 5.00000E-07\n")
        f.write("convergence gradient 5.00000E-07\n")
        f.write("xc %s  \n" % (xcf))
        f.write("end \n")
        f.write("task sodft ")
        f.write("\n")

    f.close()

# Function to retrieve energies from the output files of multiple calculations.
def get_energy_many_calc(basename, N):
    s = "grep 'Total SO-DFT' " + basename + ".out"
    process = Popen(s, shell=True, stdout=PIPE)
    stdout, stderr = process.communicate()

    nwfail = False
    err = process.returncode
    energy = []
    if err == 0:
        line = stdout.split()
        for i in range(N):
            energy.append(float(line[4 + 5 * i]))
    else:
        print("NWChem failed %s" % (str(err)))
        for i in range(N):
            energy.append(-1e10)
        nwfail = True

    return energy, nwfail
