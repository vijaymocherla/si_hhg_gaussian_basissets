import numpy as np

def parse_basisfile(path_to_basisfile):
    """Parses a basis file and returns a dictionary of atom-wise basis lines.
    """
    file = open(path_to_basisfile)
    data = file.read().split('****')
    basis_dict = {}
    for basis in data[1:-1]:
        basis_lines = basis.split("\n")
        atom = basis_lines[1].split(" ")[0]
        basis_dict[atom] = basis_lines[2:]
    return basis_dict

def get_exponents(basislines, return_exp='all'):
    """Given a list of basisset lines, parses to find exponents of each type of 
    dunning and hybrid functions. Returns a dictionary of lists based options provided. 
    """
    hybrid = False
    nfuncs = {'S':1, 'P':3, 'D':6, 'F':10, 'G':15, 'H':21}
    hybrid_exponents = {}
    dunning_exponents = {}
    contr_bool = False    
    for idx, line in enumerate(basislines):
        if len(line) > 0:
            if line[0] == '!':
                if line == '! hybrid functions':
                    hybrid = True
                continue
            if not line.split(" ")[0].isalpha():
                exp, coeff = [float(val.replace('D','E'))for val in line.split()]                    
                if hybrid:
                    if funcType in hybrid_exponents:
                        hybrid_exponents[funcType] += [exp]*nfuncs[funcType]
                    else:
                        hybrid_exponents[funcType] = [exp]*nfuncs[funcType]
                else:
                    if funcType in dunning_exponents:
                        if contr_bool:
                            dunning_exponents[funcType] = [exp]*nfuncs[funcType]     
                        dunning_exponents[funcType] += [exp]*nfuncs[funcType]
                    else:
                        dunning_exponents[funcType] = [exp]*nfuncs[funcType]
            else:
                funcType = basislines[idx].split(" ")[0]
                contraction_order = int(basislines[idx].split(" ")[4])
                if contraction_order != 1:
                    contr_bool = True
                else:
                    contr_bool = False
    exponents = {}
    if return_exp == 'hybrid':
        exponents.update(hybrid_exponents)
    elif return_exp == 'dunning':
        exponents.update(dunning_exponents)
    elif return_exp == 'all':
        exponents = dunning_exponents
        for function in hybrid_exponents:
            if function in exponents:
                exponents[function] += hybrid_exponents[function]
            else:
                exponents[function] = hybrid_exponents[function]
    else:
        raise Exception("Given `return_exp` value is invalid!")
    return exponents


def analyze_basis(path_to_basisfile, atom, hybrid_basis=True, print_min_exponents=True):
    """Analysis the basisset for a given atom in the provided basisfile.
    """
    basis_dict = parse_basisfile(path_to_basisfile)
    basislines = basis_dict[atom]
    if hybrid_basis:
        exponents = get_exponents(basislines, return_exp='hybrid')
    else:
        exponents = get_exponents(basislines, return_exp='dunning')
    if print_min_exponents:
        print('{l:2s}   {min:18s}   {max:18s}'.format(l='l', min='min (rmax a.u.)', max='max (rmin a.u)'))
        hwhm = lambda exp : np.sqrt(np.log(2)/exp)
        for funcType in exponents:
            exps = exponents[funcType]
            exp_min, exp_max = min(exps), max(exps)
            print('{l:2s}   {min:3.2E}({max_hwhm:2.2f})   {max:3.2E}({min_hwhm:2.2f})'.format(l=funcType, min=exp_min, max_hwhm=hwhm(exp_min), max=exp_max, min_hwhm=hwhm(exp_max)))

