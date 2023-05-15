#include <iostream>
#include <complex>
#include <functional>
#include <string>
#include <time.h>

#include <bar.h>
#include <fluid.h>
#include <grid.h>
#include <hamop.h>
#include <wavefunction.h>
#include <powers.hh>
#include <parameter.hh>
#include <smallHelpers.hh>

// Functions determining the potentials are defined in potentials.hh
#include <potentials.hh>

using std::cout;
using std::endl;
using std::string;

void ell_m_consistency(long ell, long m, grid g) {
  if (ell<labs(m)) {
    cerr << "|m| is greater than ell" << endl;
    exit(-1);
  };
  if (g.ngps_y()<ell+1) {
    cerr << "grid too small for ell" << endl;
    exit(-1);
  };
};
  

int main(int argc, char **argv) 
{
    clock_t t = clock();
    // Variables
    double acc, E_tot_prev;
    grid g;
    hamop hamilton;
    wavefunction staticpot, E_i;
    fprintf(stdout," --------------------------------------------------------\n");
    fprintf(stdout,"         Imaginary-time propagation initialized          \n");
    fprintf(stdout,"  (C) Copyright by Bauer D and Koval P, Heidelberg (2005)\n");
    fprintf(stdout," --------------------------------------------------------\n");

    parameterListe para("initial.param");
    double nuclear_charge = para.getDouble("nuclear-charge");

    scalarpot scalarpotx(nuclear_charge, para.getDouble("pot-cutoff"));

    //
    // Input
    //
    // *** Declare the grid ***
    g.set_dim(para.getLong("qprop-dim")); // 34 for linear polarization, 44 for polarization in the XY plane
    const double delta_r = para.getDouble("delta-r");
    g.set_ngps(long(para.getDouble("radial-grid-size")/delta_r), para.getLong("ell-grid-size"), 1);  // <--------------------------------- Max. length in r-direction, in ell-direction, always 1
    g.set_delt(delta_r);
    g.set_offs(0, 0, 0);

    int iinitmode = para.getLong("init-type"); // 1 -- random, 2 -- hydrogenic wf. 

    int iv        = 1; // Verbosity of stdout
        
    //
    // Prepare for propagation ...
    // 

    // Number of imaginary time steps
    const long lno_of_ts=para.getLong("im-time-steps");
    fluid ell_init, m_init;
    ell_init.init(g.ngps_z());
    m_init.init(g.ngps_z());

    const long nr = para.getLong("initial-nr");
    const long my_m_quantum_num=para.getLong("initial-m");
    const long my_ell_quantum_num=para.getLong("initial-ell");
    ell_m_consistency(my_ell_quantum_num, my_m_quantum_num, g);
    wavefunction wf[nr+1];
    ell_init[0] = my_ell_quantum_num; // Populated l quantum number (needed for initialization only)  <---------------------------- 1s, 2p, 3d, ?????
    m_init[0] = my_m_quantum_num;     // Populated m quantum number (needed for initialization only)
    const int me = 0; // Dummy here

    E_i.init(g.ngps_z());

    // Set the purly imaginary timestep
    const double imag_timestep=0.25*g.delt_x();

    // The Hamiltonian
    imagpot imaginarypot(0.0, 0.0, 0.2, 0.2);
    hamilton.init(g, always_zero2, always_zero2, always_zero2, scalarpotx, always_zero5, always_zero5, imaginarypot, always_zero2);

    // This is the linear and constant part of the Hamiltonian
    staticpot.init(g.size()); 
    staticpot.calculate_staticpot(g, hamilton);

    // Create some files with appropriate appendices
    string common_prefix("./dat/imag_prop_");
    string str_fname_logfi=common_prefix+to_string(my_m_quantum_num)+string(".log");
    FILE* file_logfi = fopen_with_check(str_fname_logfi, "w");

    string str_fname_obser=common_prefix+to_string(my_m_quantum_num)+string("_observ.dat");
    FILE* file_obser_imag = fopen_with_check(str_fname_obser,"w");

    string str_fname_wf_ini=common_prefix+to_string(my_m_quantum_num)+string("_wf_ini.dat");
    FILE* file_wf_ini = fopen_with_check(str_fname_wf_ini,"w");

    string str_fname_wf_fin=common_prefix+to_string(my_m_quantum_num)+string("_wf_fin.dat");
    FILE* file_wf_fin = fopen_with_check(str_fname_wf_fin,"w");
    
    string str_fname_potent=common_prefix+to_string(my_m_quantum_num)+string("_potential.dat");
    FILE* file_potent = fopen_with_check(str_fname_potent, "w");

    if (iv!=0) {
    cout  <<  str_fname_logfi  << " will be (re)written." << endl
            <<  str_fname_wf_ini  << " will be (re)written." << endl
            <<  str_fname_obser  << " will be (re)written." << endl
            <<  str_fname_wf_fin  << " will be (re)written." << endl
            <<  str_fname_potent  << " will be (re)written." << endl;
    };

    // *** New initialization ***
    // *** The wavefunction array
    for (long nr1=0; nr1<nr+1; nr1++) 
    {
        wf[nr1].init(g.size());
        if (g.dimens()==34) 
        {
            wf[nr1].init(g, iinitmode, 1.0, ell_init);
        }
        else if (g.dimens()==44) 
        {
            wf[nr1].init_rlm(g, iinitmode, 1.0, ell_init, m_init);
        };
        wf[nr1].normalize(g);
    };


    fprintf(stdout, "Norm of KS-orbital: %le\n", wf[0].norm(g));

    // Write in log file
    fprintf(file_logfi,"Imaginary-time propagation\n");
    fprintf(file_logfi,"Grid: \n");
    fprintf(file_logfi,"g.ngps_x() = %ld\n", g.ngps_x());
    fprintf(file_logfi,"g.ngps_y() = %ld\n", g.ngps_y());
    fprintf(file_logfi,"g.ngps_z() = %ld\n", g.ngps_z());
    fprintf(file_logfi,"g.dimens() = %d\n\n", g.dimens());
    fprintf(file_logfi,"g.delt_x() = %20.15le\n", g.delt_x());

    fprintf(file_logfi,"imag_timestep     = %20.15le\n", imag_timestep);
    fprintf(file_logfi,"lno_of_ts         = %ld\n", lno_of_ts);
    fprintf(file_logfi,"nuclear_charge    = %20.15le\n", nuclear_charge);
    fprintf(file_logfi,"iinitmode         = %d\n", iinitmode);
    fprintf(file_logfi,"str_fname_wf_ini = %s\n", str_fname_wf_ini.c_str());
    fprintf(file_logfi,"str_fname_obser  = %s\n", str_fname_obser.c_str());
    fprintf(file_logfi,"str_fname_wf_fin = %s\n", str_fname_wf_fin.c_str());
    fflush(file_logfi);

    const double dr  = g.delt_x();
    for (long ir=0; ir<g.ngps_x(); ir++) 
    {
        const double R = double(ir+1)*dr;
        // calculate the total energy
        const double U = hamilton.scalarpot(R,my_ell_quantum_num,1.0,0.0,0.0);
        fprintf(file_potent,"%20.15le %20.15le \n", R, U);
    };
    fclose(file_potent);

    // *** The orbital before relaxation
    wf[0].dump_to_file_sh(g, file_wf_ini, 1);
    fclose(file_wf_ini);

    // ********************************************************
    // ***** Imaginary propagation ****************************
    // ********************************************************
    double E_tot;
    const cplxd timestep(0.0, -1.0*imag_timestep);
    for (long nr1=0; nr1<=nr; nr1++) 
    {
        E_tot = 0.0;
        for (long ts=0; ts<lno_of_ts; ts++) 
        {
            if (ts%1000==0)
                cout << ts << endl;
            const double time = double(ts)*imag(timestep);
//             calculate the total energy
            E_tot_prev = E_tot;
            E_tot = real(wf[nr1].energy(0.0, g, hamilton, me, staticpot, nuclear_charge));
            acc = fabs((E_tot_prev-E_tot)/(E_tot_prev+E_tot));
            fprintf(file_obser_imag,"% 20.15le % 20.15le % 20.15le %ld\n", time, E_tot, acc, ts);
            if (ts%1000==0)
                fprintf(stdout,"% 20.15le % 20.15le\n\n", E_tot, acc);
            for (long nr2=0; nr2<nr1; nr2++)
            {
                wf[nr1].subtract(g, wf[nr2]);
            };
            wf[nr1].propagate(timestep, 0.0, g, hamilton, me, staticpot, my_m_quantum_num, nuclear_charge);
            wf[nr1].extract_ell_m(g, my_ell_quantum_num, my_m_quantum_num);
            wf[nr1].normalize(g);

        };
        fprintf(stdout,"% 20.15le % 20.15le\n\n", E_tot, acc);
    };

    
    fprintf(stdout," %ld imaginary steps performed.\n", lno_of_ts);
    fprintf(stdout, "     E_tot                  accuracy \n");
    fprintf(stdout,"% 20.15le % 20.15le\n\n", E_tot, acc);
    fprintf(file_logfi, "acc = %le\n", acc);

    
    fclose(file_obser_imag);
    fclose(file_logfi);

    wf[nr].dump_to_file_sh(g,file_wf_fin,1);
    fclose(file_wf_fin);

    if (iv!=0)
    {
        cout  <<  str_fname_logfi  << " is written." << endl
            <<  str_fname_wf_ini  << " is written." << endl
            <<  str_fname_obser  << " is written." << endl
            <<  str_fname_wf_fin  << " is written." << endl;
    };
    float sec = ((float)(clock() - t))/CLOCKS_PER_SEC;
    cout << " Timer: " << long(sec)/3600 << "h " << long(sec)%3600/60 << "min " << (long(sec*100)%6000)/100.0 << "sec.\n";
//     fprintf(stdout, "Hasta la vista...\n");
    fprintf(stdout," -------------------------------------------------------\n");
    fprintf(stdout,"       Imaginary-time propagation finished        \n");
    fprintf(stdout," -------------------------------------------------------\n");
}; // End of main program
