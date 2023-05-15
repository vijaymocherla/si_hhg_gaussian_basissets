Example: Hydrogen, lambda=800 nm, I=10^14 Wcm^-2, linear polarization along Z-axis

Quickstart:
./do_all.sh

or 

Step-by-step:
make clean
make imag_prop
./imag_prop
make real_prop
./real_prop

Remarks:
"make clean" leads to a fresh compilation of the qprop library. If you run into problems during compilation or execution try this first.

"make imag_prop" builds the program for calculationg the initial state by imaginary propagation

"make real_prop" builds the program for the real time propagtion.

To perform the Fourier transform of the results and visualize the HHG spectrum and/or the average dipole acceleration, use the Python script "plot_hhg.py"

To visualize the laser time dependence and/or geometry use the Python script "plot_laser.py"