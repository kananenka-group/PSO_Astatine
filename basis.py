# Function to save a basis set to a file
def save(basis, filename):
    f = open(filename, "w")
    N = len(basis)
    f.write(" %d \n" % (N))  # Write the number of basis functions (N)
    for bas in basis:
        Nb = len(bas)
        f.write(" %d \n" % (Nb))  # Write the number of basis functions in this shell (Nb)
        for x in bas:
            f.write(" %s   %15.9f \n" % (x[0], x[1]))  # Write the shell type (sym) and exponent (val)
    f.close()
    return

# Function to read a basis set from a file
def read(filename):
    f = open(filename, "r")
    line = f.readline()
    line2 = line.split()
    N = int(line2[0])  # Read the number of basis functions (N)

    basis = []

    for nx in range(N):
        line = f.readline()
        line2 = line.split()
        nb = int(line2[0])  # Read the number of basis functions in this shell (nb)
        bas = []
        for ny in range(nb):
            line = f.readline()
            line2 = line.split()
            sym = line2[0]  # Read the shell type (sym)
            val = float(line2[1])  # Read the exponent (val)
            exp = []
            exp.append(sym)
            exp.append(val)
            bas.append(exp)
        basis.append(bas)

    f.close()
    return basis

# Function to write a basis set to a file in a specific format
def write_basis(bas, gen, idx):
    filename = "./basis/bas_gen" + str(gen) + "_id" + str(idx) + ".dat"
    f = open(filename, "w")
    for p in range(len(bas)):
        v = bas[p]
        f.write("   %s     %18.9f \n" % (v[0], v[1]))  # Write shell type (v[0]) and exponent (v[1])
    f.close()
    return

# Function to read a starting basis set from a file
def read_start_basis(start_file):
    f = open(start_file, "r")
    basis = []
    for line in f:
        ls = []
        line2 = line.split()
        sh_ = line2[0]  # Read shell type
        ex_ = float(line2[1])  # Read exponent
        ls.append(sh_)
        ls.append(ex_)
        basis.append(ls)
    f.close()
    return basis

# Function to read a pseudopotential from a file
def read_pseudopotential(pp_file):
    """
    Since the pseudopotential part
    will be left untouched will simply
    read it here
    """
    f = open(pp_file, "r")
    pp = f.read()
    f.close()
    return pp

# Function to read an atom basis from a file
def read_atom_basis(basis_file):
    f = open(basis_file, "r")
    bs = f.read()
    f.close()
