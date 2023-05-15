// Definitions of potentials

double always_zero2(double t, int me) {
    return 0;
};

double always_zero5(double x, double y, double z, double t, int me) {
    return 0;
};

// The vector potential with sine squared envelope
class vecpot 
{
    double omega1;
    double omega2;
    double  E_1;
    double  E_2;
    double   n_c1;
    double   n_c2;
    double    phi_cep1;
    double    phi_cep2;
    double     tau;
    double       ww1;
    double       ww2;
    double  duration1;
    double  duration2;
    double  duration;
public:
    vecpot(double om1, double om2, double n_cyc1, double n_cyc2, double delay, double E_max1, double E_max2, double cep1, double cep2) : omega1(om1), omega2(om2), n_c1(n_cyc1), n_c2(n_cyc2), tau(delay), E_1(E_max1), E_2(E_max2), phi_cep1(cep1), phi_cep2(cep2)
    {
        duration1=n_c1*2*M_PI/omega1;
        duration2=n_c2*2*M_PI/omega2+tau;
        duration = (duration1>duration2)?duration1:duration2; // Total duration of the (overlaping) pulses
        // Angular frequency of the envelopes
        ww1=omega1/(2.0*n_c1);
        ww2=omega2/(2.0*n_c2);
    };
    double operator()(double time, int me) const 
    {
        double result = 0.0;
        if (time<duration1)
        {
            // Here's the shape of the laser pulse
            if (E_1!=0.0)
            {
                if (time<=2*2*M_PI/omega1)
                    result += E_1/omega1 * pow2(sin(0.125*omega1*time))*sin(omega1*time + phi_cep1);
                else if (time<(n_c1-2.0)*2*M_PI/omega1)
                    result += E_1/omega1 * sin(omega1*time + phi_cep1);
                else
                    result += E_1/omega1 * pow2(sin(0.125*(omega1*time-(n_c1-4.0)*2*M_PI)))*sin(omega1*time + phi_cep1);
            };
        };
        if ((time>tau)&&(time<duration2))
        {
            if (E_2!=0.0)
            {
               // if (time<=tau+2*2*M_PI/omega2)
               //     result += E_2/omega2*pow2(sin(0.125*omega2*(time-tau)))*sin(omega2*(time-tau) + phi_cep2);
               // else if (time<tau+(n_c2-2.0)*2*M_PI/omega2)
               //     result += E_2/omega2 * sin(omega2*(time-tau) + phi_cep2);
               // else
               //     result += E_2/omega2 * pow2(sin(0.125*(omega2*(time-tau)-(n_c2-4.0)*2*M_PI)))*sin(omega2*(time-tau) + phi_cep2);

                result += -E_2/(omega2*(2-2/pow2(n_c2)))*((-1/pow2(n_c2))+ (-1+1/pow2(n_c2)+cos(omega2*time/n_c2))*cos(omega2*time) + (1/n_c2)*sin(omega2*time/n_c2)*sin(omega2*time));
            };
        };
        return result;
    };
    double get_duration() 
    {
        return duration;
    };
    double get_Up() 
    {
        return 0.25*(E_1*E_1)/(omega1*omega1);
    };

};

class scalarpot 
{
  double nuclear_charge;
  double R_co;
public:
  scalarpot(double charge, double co) : nuclear_charge(charge), R_co(co) {
    //
  };
  double operator()(double x, double y, double z, double time, int me) const {

    // Scrinzi potential
    // double result=(x<R_co)?nuclear_charge*(-1.0/x-pow2(x)/(2*pow3(R_co))+3.0/(2.0*R_co)):0.0;
    // result =  nuclear_charge*result;
    // --------------------------------------------------------------------
    // Simple Volker potential; first -1/r then linear then zero
    // const double m=1.0/pow2(R_co);
    // double result=(x<R_co)?-1.0/x:((x<2*R_co)?-1.0/R_co+m*(x-R_co):0.0);
    // result =  nuclear_charge*result;
    // --------------------------------------------------------------------
    // V_SAE fitting paramters for ground state.
    //  # He  (Co, Zc) = (1,1); c = 2.135000 
    //  # H-  (Co, Zc) = (0,1); c = 0.880870
    //  # Li+ (Co, Zc) = (2,1); c = 3.421875 
    double Co = 1.0;
    double Zc = 1.0;
    double c = 2.135000; 
    double result = -(Co+Zc*exp(-c*x))/x;
    return result;
  };
  double get_nuclear_charge() {return nuclear_charge; };
};

class imagpot {
  //long imag_potential_width;
  double imag_ampl;  // Amplitude of imaginary absorbing potential  <--------------  100.0 imag pot on,  0.0 off
  double imag_start;
  double man_k;
  double man_d;
public:
  imagpot(double ampl, double start, double cap_k, double cap_d) : imag_ampl(ampl), imag_start(start), man_k(cap_k), man_d(cap_d) {
    //
  };
  double operator()(long xindex, long yindex, long zindex, double time, grid g) {

 //   if (ampl_im>1.0) {
 //     const long imag_start=g.ngps_x()-imag_potential_width;
 //     if (xindex<imag_start)
 //	return 0;
 //     else {
 // const double r=double(xindex-imag_start)/double(imag_potential_width);
 // return ampl_im*pow2(pow2(pow2(pow2(r))));
 //     };
 //   }
 //   else {
 //     return 0.0;
 //   };

    const double man_c = 2.6220575542921198105;
    const double man_a = 1-16/pow3(man_c);
    const double man_b = (1-17/pow3(man_c))/pow2(man_c);
    const double man_e = pow2(man_k)/2;

    if (imag_ampl>1.0) {
      const double x = g.r(xindex);
      if (x<imag_start)
        return 0;
      else {
        const double r = 2*man_d*man_k*(x-imag_start);
        return man_e * (man_a*r - man_b*pow3(r) + 4/pow2(man_c-r) - 4/pow2(man_c+r));
      };
    }
    else {
      return 0.0;
    };
  };
};
