"""
This is the Stellar_core module of PyMoS4 package whose objective is to simulate 
the internal structure of a static star model.

Gabriel Wendell et al 2021.
"""

import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

plt.rcParams['font.family'] = 'serif'


# ---------------------------------------------------------------------------------------------------------------------
def opt_plot():
    """
    This function helps to improve the aesthetics of the graphics!
    """
    # plt.style.use('dark_background')
    plt.grid(True, linestyle=':', color='0.50')
    plt.minorticks_on()
    plt.tick_params(axis='both',which='minor', direction = "in",
                        top = True,right = True, length=5,width=1,labelsize=15)
    plt.tick_params(axis='both',which='major', direction = "in",
                        top = True,right = True, length=8,width=1,labelsize=15)


# ---------------------------------------------------------------------------------------------------------------------
"""
Here you can turn the dynamic step size 'On' or 'Off'.
"""
dynamic_steplength = 'On'

# ---------------------------------------------------------------------------------------------------------------------
def read_kappa(T, rho):
    """
    This functions read the Opactities.txt file, 
    and interpolates or extrapolates its values if necessary.
    """

    rho_cgs = 1e-3*rho
    R = 1e18*rho_cgs*T**(-3.)
    infile = open('Opacities.txt')
    rows = infile.readlines()
    infile.close()
    log_R = rows[0].split()
    log_R = log_R[1:]
    log_R = [float(logR) for logR in log_R]

    log_T = []
    log_kappa = []
    # Skip log_R row and a blank row
    for r in rows[2:]:
        r_list = r.split()
        r_list = [float(rs) for rs in r_list]
        temperature = r_list[0]
        kappas = r_list[1:]
        log_T.append(temperature)
        log_kappa.append(kappas)

    # [log_T] = K
    log_T = np.array(log_T)
    # cgs system
    log_R = np.array(log_R)
    # cgs system
    log_kappa = np.array(log_kappa)

    T_log = np.log10(T)
    R_log = np.log10(R)

    # When the input value can be found in the table:
    for i in range(len(log_T)):
        if T_log == log_T[i]:
            kappa_1index = i
            print('log(T) was found in the table!')
            break
        else: 
            kappa_1index = 'Not found'

    for i in range(len(log_R)):
        if R_log == log_R[i]:
            kappa_2index = i
            print('log(R) was found in the table!')
            break
        else: 
            kappa_2index = 'Not found'
    
    # Extrapolation warnings!
    if T_log < 3.75 or T_log > 8.7:
        print('Extrapolation: log(T) outside the bounds of the table!')
    if R_log < -8.0 or R_log > 1:
        print('Extrapolation: log(R) outside the bounds of the table!')



    if kappa_1index != 'Not found' and kappa_2index != 'Not found':
        log_kappa_value = log_kappa[kappa_1index][kappa_2index]
        # cgs system
        kappa_value = 10**log_kappa_value
    else:
        log_kappa_value_function = interpolate.interp2d(log_R, log_T, log_kappa, kind = 'linear', copy = False, bounds_error = False, fill_value = None)
        log_kappa_value = log_kappa_value_function(R_log, T_log)
        # cgs system
        kappa_value = 10**log_kappa_value[0]

    # kappa (S.I.)
    return kappa_value / 10.


# ---------------------------------------------------------------------------------------------------------------------
def energy_PP(T, rho):
    """
    This function returns the energy generation through PP chain reactions
    """
    # Fractional abundances by weight
    X = 0.7       # H 
    Y_3 = 1e-10   # He_3
    Y = 0.29      # He_3 + He_4
    Z = 0.01      # Metals 
    Z_Li = 1e-7   # Li
    Z_Be = 1e-7   # Be

    m_u = 1.6605*10**(-27.)                # [m_u] = kg
    MeV = 1.60218*10**(-13.)               # [MeV] = J
    Q_pp = (0.15 + 1.02)*MeV               # [Q_pp] = J
    Q_Dp = 5.49*MeV                        # [Q_Dp] = J
    Q_33 = 12.86*MeV                       # [Q_33] = J
    Q_34 = 1.59*MeV                        # [Q_34] = J
    Q_7e = 0.05*MeV                        # [Q_7e] = J
    Q_71_ = 17.35*MeV                      # [Q_71_] = J
    Q_71 = (0.14 + 1.02 + 6.88 + 3.0)*MeV  # [Q_71] = J
   
    T_9 = T*10**(-9.)          # Temperature conversion to units 10^9 K
    T_9_ = T_9/(1+4.95*10**(-2.)*T_9)
    T_9__ = T_9/(1+0.759*T_9)
    N_A = 6.022*10**23         # [N_A] = 1/mol

    N_A_Lamb_pp = 4.01*10**(-15.)*T_9**(-2./3.)*np.exp(-3.38*T_9**(-1./3.))*(1. + 0.123*T_9**(1./3.) + 1.09*T_9**(2./3.) + 0.938*T_9)
    N_A_Lamb_33 = 6.04e10*T_9**(-2./3.)*np.exp(-12.276*T_9**(-1./3.))*(1+0.034*T_9**(1./3.) - 0.522*T_9**(2./3.) - 0.124*T_9 + 0.353*T_9**(4./3.) + 0.213*T_9**(-5./3.))
    N_A_Lamb_34 = 5.61e6*T_9_**(5./6.)*T_9**(-3./2.)*np.exp(-12.826*T_9_**(-1./3.))
    N_A_Lamb_7e = 1.34*10**(-10.)*T_9**(-1./2.)*(1. - 0.537*T_9**(1./3.) + 3.86*T_9**(2./3.) + 0.0027*T_9**(-1.)*np.exp(2.515*10**(-3.)*T_9**(-1.)))
    N_A_Lamb_71_ = 1.096e9*T_9**(-2./3.)*np.exp(-8.472*T_9**(-1./3.)) - 4.83e8*T_9__**(5./6.)*T_9**(-3./2.)*np.exp(-8.472*T_9__**(-1./3.)) + 1.06e10*T_9**(-3./2.)*np.exp(-30.442*T_9**(-1.))
    N_A_Lamb_71 = 3.11e5*T_9**(-2./3.)*np.exp(-10.262*T_9**(-1./3.)) + 2.53e3*T_9**(-3./2.)*np.exp(-7.306*T_9**(-1.))

    n_p = rho*X/(1.*m_u)
    n_He = rho*Y/(4.*m_u)       # Both He_4 and He_3
    n_He_3 = rho*Y_3/(3.*m_u)
    n_He_4 = n_He - n_He_3
    n_Be = rho*Z_Be/(7.*m_u)
    n_Li = rho*Z_Li/(7.*m_u)
    n_e = n_p + 2.*n_He_3 + 2.*n_He_4 + 2.*n_Be + 1.*n_Li

    Lamb_pp = 1e-6*N_A_Lamb_pp/N_A  # [Lamb_pp] = m^3/s  
    Lamb_33 = 1e-6*N_A_Lamb_33/N_A  # [Lamb_33] = m^3/s
    Lamb_34 = 1e-6*N_A_Lamb_34/N_A  # [Lamb_34] = m^3/s
    if T < 1e6:
        if N_A_Lamb_7e > 1.57e-7/n_e:
           N_A_Lamb_7e = 1.57e-7/n_e
    Lamb_7e = 1e-6*N_A_Lamb_7e/N_A    # [Lamb_7e] = m^3/s  ; (Be + e)
    Lamb_71_ = 1e-6*N_A_Lamb_71_/N_A  # [Lamb_71_] = m^3/s ; (Li + H)
    Lamb_71 = 1e-6*N_A_Lamb_71/N_A    # [Lamb_71] = m^3/s  ; (Be + H)


    r_pp = n_p*n_p*Lamb_pp/(rho*2.)
    r_33 = n_He_3*n_He_3*Lamb_33/(rho*2.)
    r_34 = n_He_3*n_He_4*Lamb_34/rho
    if r_pp < (r_33*2. + r_34):
        rate1 = r_pp/(2.*r_33 + r_34)
        r_33 *= rate1
        r_34 *= rate1
        
    r_7e = n_Be*n_e*Lamb_7e/rho
    r_71_ = n_Li*n_p*Lamb_71_/rho
    r_71 = n_Be*n_p*Lamb_71/rho
    if r_34 < (r_7e + r_71):
        rate2 = r_34/(r_7e + r_71)
        r_7e *= rate2
        r_71 *= rate2
    if r_7e < r_71_:
        rate3 = r_7e/r_71_
        r_71_ *= rate3

    # return r_pp*(Q_pp + Q_Dp)*rho, r_33*Q_33*rho, r_34*Q_34*rho, r_7e*Q_7e*rho, r_71_*Q_71_*rho, r_71*Q_71*rho
    eps = r_pp*(Q_pp + Q_Dp) + r_33*Q_33 + r_34*Q_34 + r_7e*Q_7e + r_71_*Q_71_ + r_71*Q_71

    return eps


# ---------------------------------------------------------------------------------------------------------------------
def calculate_P(rho, T):
    """
    This function calculates the pressure inside the star
    """
    sigma = 5.67e-8          # [simga] = W * m^-2 * K^-4
    c = 2.998e8              # [c] = m/s
    k_B = 1.382e-23          # [k_B] = m^2 * kg * s^-2 * K^-1
    m_u = 1.6605*10**(-27.)  # [m_u] = kg
    a = (4.*sigma)/c
    Z = 0.01
    X = 0.7
    Y = 0.29
    mu = 1./(2*X + 2.*Y/4. + Z/2.)
    P_rad = (a*T**4.)/3.
    P_gas = rho*k_B*T/(mu*m_u)
    P = P_rad + P_gas

    return P


# ---------------------------------------------------------------------------------------------------------------------
def calculate_rho(P,T): 
    """
    This function calculates the density along the internal structure of the star
    """
    sigma = 5.67e-8          # [simga] = W * m^-2 * K^-4
    c = 2.998e8              # [c] = m/s
    k_B = 1.382e-23          # [k_B] = m^2 * kg * s^-2 * K^-1
    m_u = 1.6605*10**(-27.)  # [m_u] = kg
    a = (4.*sigma)/c
    Z = 0.01
    X = 0.7
    Y = 0.29
    mu = 1./(2.*X + 3.*Y/4. + Z/2.)
    P_rad = (a*T**4.)/3.   
    P_gas = P - P_rad
    rho = P_gas*mu*m_u/(k_B*T)

    return rho


# ---------------------------------------------------------------------------------------------------------------------
"""
Some important constants...
"""
# Radius of the Sun
R_Sun = 6.96e8     # [R_Sun] = m
# Average density of the Sun
rho_Sun = 1.408e3  # [rho_Sun] = kg/m^3
# Mass of the entire Sun
M_Sun = 1.989e30   # [M_Sun] = kg
# Luminosity of the Sun 
L_Sun = 3.846e26   # [L_Sun] = W
# Universal gravitational constant
G = 6.672e-11      # [G] = N * (m/kg)^2
# Stefan-Boltzmann constant
sigma = 5.67e-8    # [simga] = W * m^-2 * K^-4

"""
Starting point : Best-fit parameters
"""
rho_0 = 5.9e3   # [rho_0] = kg/m^3
T_0 = 8.6e6     # [T_0] = K
R_0 = 0.3711*R_Sun
M_0 = 0.8*M_Sun

if dynamic_steplength == 'Off':
    M_final = 0.
    # Here, the fixed steplength is going to vary:
    dm = -1e-4*M_Sun
    # Number of datapoints:
    n = 1 + (M_0 - M_final)/abs(dm)
    n = int(n)
    mass = np.linspace(M_0, M_final, n)
else:
    # Here you define the maximun number of data points,
    # if the mass is negative or the steplength becomes too small
    # the code cuts the loop and traces the plot before that
    n = 1e4
    n = int(n)
    mass = np.zeros(n)
    mass[0] = M_0

radius = np.zeros(n)
pressure = np.zeros(n)
luminosity = np.zeros(n)
temperature = np.zeros(n)
density = np.zeros(n)
epsilon = np.zeros(n)

radius[0] = R_0
pressure[0] = calculate_P(rho_0, T_0)
luminosity[0] = L_Sun
temperature[0] = T_0
density[0] = rho_0
epsilon[0] = energy_PP(T_0, rho_0)

# Plot the whole array if the mass doesn't get negative:
breaking_point = n

print_counter = 0.
for i_ in range(1, n):
    if dynamic_steplength == 'Off':
        radius[i_] = radius[i_-1] + dm*(1./(4.*np.pi*(radius[i_-1]**2)*density[i_-1]))
        pressure[i_] = pressure[i_-1] + dm*((-G*mass[i_-1])/(4.*np.pi*(radius[i_-1]**4)))
        luminosity[i_] = luminosity[i_-1] + epsilon[i_-1]*dm
        kappa = read_kappa(temperature[i_-1], density[i_-1])
        temperature[i_] = temperature[i_-1] + (-3.*kappa*luminosity[i_-1]/(256.*np.pi**2*sigma*radius[i_-1]**4.*temperature[i_-1]**3))*dm
        density[i_] = calculate_rho(pressure[i_], temperature[i_])
        epsilon[i_] = energy_PP(temperature[i_],density[i_])
    else:
        # Allowed fraction of change: 
        p = 0.1 
        f1 = 1./(4.*np.pi*(radius[i_-1]**2)*density[i_-1])
        dm1 = p*radius[i_-1]/f1
        f2 = -G*mass[i_-1]/(4.*np.pi*(radius[i_-1]**4))
        dm2 = p*pressure[i_-1]/f2
        f3 = epsilon[i_-1]
        dm3 = p*luminosity[i_-1]/f3
        kappa = read_kappa(temperature[i_-1], density[i_-1])
        f4 = -3.*kappa*luminosity[i_-1]/(256.*np.pi**2*sigma*radius[i_-1]**4.*temperature[i_-1]**3)
        dm4 = p*temperature[i_-1]/f4
        dm = -1.*(min(abs(dm1), abs(dm2), abs(dm3), abs(dm4)))
        mass[i_] = mass[i_-1] + dm
        radius[i_] = radius[i_-1] + dm*f1
        pressure[i_] = pressure[i_-1] + dm*f2
        luminosity[i_] = luminosity[i_-1] + f3*dm
        temperature[i_] = temperature[i_-1] + f4*dm
        density[i_] = calculate_rho(pressure[i_], temperature[i_])
        epsilon[i_] = energy_PP(temperature[i_],density[i_])

    if mass[i_] < 0.:
        # Cut the loop if the mass turns negative:
        breaking_point = i_ 
        break
    # Break before the step gets sufficient low that the majority of values is around the final value:
    if abs(dm) < 1e-6*M_0: 
        breaking_point = i_ 
        break
    print_counter += 1.
   # Print every 10th values:
    if print_counter > 10.: 
        print('##################')
        print(' ')
        print('EVOLUTION OF EACH PARAMETER:')
        print(' ')
        print('dm =', dm)
        print('R/R_Sun:', radius[i_]/R_Sun)
        print('M/M_Sun:', mass[i_]/M_Sun)
        print('L/L_Sun:', luminosity[i_]/L_Sun)
        print('rho/rho_Sun:', density[i_]/rho_Sun)
        print('T:', temperature[i_])
        print('P:', pressure[i_])
        print('epsilon:', epsilon[i_])
        print('kappa:', kappa)
        print(' ')
        print('##################')
        print(' ')
        print_counter = 0.

print('Number of datapoints used:',  breaking_point)

# Logarithmic Scale
l_eps = np.log10(epsilon[:breaking_point])
l_pressure = np.log10(pressure[:breaking_point])
l_density = np.log10(density[:breaking_point])

# Check the radius of the core
for l in range(len(luminosity)):
   if luminosity[l]/L_Sun < 0.995:
      core_starts = l
      print('Radius of the core:', (radius[l]/radius[0])*100, '% of initial radius.')
      break
      

# Print the final values of each stellar parameter
print(' ')
print(mass[breaking_point-1], radius[breaking_point-1], luminosity[breaking_point-1])
print(' ')
print('Final values of each stellar parameter:')
print(' ')
print('Final Radius:', (radius[breaking_point-1]/radius[0])*100, '%')
print('Final Mass:', (mass[breaking_point-1]/mass[0])*100, '%')
print('Final Luminosity:',(luminosity[breaking_point-1]/luminosity[0])*100, '%')
print(' ')



# ---------------------------------------------------------------------------------------------------------------------
"""
From this point onwards, the construction of the final plots is made!
"""
plt.figure(figsize = (10,8))
plt.plot(radius[:breaking_point]/R_Sun, l_eps, color='indigo', label = 'Energy Production')
plt.title('Energy Production vs. Radius', fontsize=20)
plt.xlabel('R/R$_{\odot}$', fontsize=18)
plt.ylabel('$\log{(\epsilon)}$ [J $\cdot$ kg$^{-1}$ s$^{-1}$]', fontsize=18)
plt.axis([radius[breaking_point-1]/R_Sun, radius[0]/R_Sun, l_eps[0], l_eps[breaking_point-1]])
opt_plot()
plt.savefig('Final_epsilon.png')
plt.show()

# If you want to use plt.style('dark_background') from the opt_plot() function you will need to run the first image twice.
# This is necessary because Matplotlib's 'dark_background' style is only applied from the second plot onwards.
# If you want to use the default Matplotlib style, uncomment the """ and turn this second plot into a comment.
"""
plt.figure(figsize = (10,8))
plt.plot(radius[:breaking_point]/R_Sun, l_eps, color='indigo', label='Energy Production')
plt.title('Energy Production vs. Radius', fontsize=20)
plt.xlabel('$R/R_{\odot}$', fontsize=18)
plt.ylabel('$\log{(\epsilon)}$ [J kg$^{-1}$ s$^{-1}$]$', fontsize=18)
plt.axis([radius[breaking_point-1]/R_Sun, radius[0]/R_Sun, l_eps[0], l_eps[breaking_point-1]])
opt_plot()
plt.savefig('Final_epsilon.png')
plt.show()
"""


plt.figure(figsize = (10,8))
plt.plot(radius[:breaking_point]/R_Sun, l_pressure, color='sienna', label='Pressure')
plt.title('Pressure vs. Radius', fontsize=20)
plt.axis([radius[breaking_point-1]/R_Sun, radius[0]/R_Sun, l_pressure[0], l_pressure[breaking_point-1]])
plt.xlabel('R/R$_{\odot}$', fontsize=18)
plt.ylabel('$\log{(P)}$ [Pa]', fontsize=18)
opt_plot()
plt.savefig('Final_Pressure.png')
plt.show()


plt.figure(figsize = (10,8))
plt.plot(radius[:breaking_point]/R_Sun, mass[:breaking_point]/M_Sun, color='black', label='Mass')
plt.scatter(radius[:breaking_point]/R_Sun, mass[:breaking_point]/M_Sun, color='royalblue')
plt.title('Mass vs. Radius', fontsize=20)
plt.axis([radius[breaking_point-1]/R_Sun, radius[0]/R_Sun, 0,0.85])
plt.xlabel('R/R$_{\odot}$', fontsize=18)
plt.ylabel('M/M$_{\odot}$', fontsize=18)
opt_plot()
plt.savefig('Final_Mass.png')
plt.show()


plt.figure(figsize = (10,8))
plt.plot(radius[:breaking_point]/R_Sun, luminosity[:breaking_point]/L_Sun, color='crimson', label='Luminosity')
plt.title('Luminosity vs. Radius', fontsize=20)
plt.xlabel('R/R$_{\odot}$', fontsize=18)
plt.ylabel('L/L$_{\odot}$', fontsize=18)
plt.axis([radius[breaking_point-1]/R_Sun, radius[0]/R_Sun, 0,1])
plt.axvline(x=radius[core_starts]/R_Sun, linestyle='dashed', color='mediumvioletred')
opt_plot()
plt.savefig('Final_Luminosity.png')
plt.show()


plt.figure(figsize = (10,8))
plt.plot(radius[:breaking_point]/R_Sun, l_density, color='mediumblue', label='Density')
plt.title('Density vs. Radius', fontsize=20)
plt.xlabel('R/R$_{\odot}$', fontsize=18)
plt.ylabel('$\log{(\\rho)}$ [kg $\cdot$ m$^{-3}$]', fontsize=18)
plt.axis([radius[breaking_point-1]/R_Sun, radius[0]/R_Sun, l_density[0],l_density[breaking_point-1]])
opt_plot()
plt.savefig('Final_Density.png')
plt.show()


plt.figure(figsize = (10,8))
plt.plot(radius[:breaking_point]/R_Sun, temperature[:breaking_point]/1e6, color='navy', label='Temperature')
plt.scatter(radius[:breaking_point]/R_Sun, temperature[:breaking_point]/1e6, color='darkcyan')
plt.title('Temperature vs. Radius', fontsize=20)
plt.axis([radius[breaking_point-1]/R_Sun, radius[0]/R_Sun, T_0/1e6,temperature[breaking_point-1]/1e6])
plt.xlabel('R/R$_{\odot}$', fontsize=18)
plt.ylabel('T [MK]', fontsize=16)
opt_plot()
plt.savefig('Final_Temperature.png')
plt.show()


# ---------------------------------------------------------------------------------------------------------------------
"""
Here we perform a summary with the initial values and the final values of the relevant stellar parameters.
"""
print('#######################')
print(' ')
print('# SUMMARY')

# Initial value of the pressure and the epsilon:
print(' ')
print('- INITIAL VALUES:')
print(' ')
print('Pressure:', calculate_P(rho_0, T_0))
print('epsilon:', epsilon[0])

# Final value of the stellar parameters:
print(' ')
print('- FINAL VALUES:')
print(' ')
print('Temperature:', temperature[breaking_point-1])
print('Luminosity', luminosity[breaking_point-1])
print('Pressure:', pressure[breaking_point-1])
print('Density:', density[breaking_point-1])
print('epsilon:', epsilon[breaking_point-1])
print(' ')

print('End of first part!')
print(' ')


# ---------------------------------------------------------------------------------------------------------------------