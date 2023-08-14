#!/usr/bin/env python
# coding: utf-8

import astropy.units as u
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Grid
import numpy as np
import os
import sys
from multiprocessing import Pool
import uuid
from tqdm import tqdm
import warnings
import subprocess

## get path of current directory
LocalPath = os.getcwd() + "/"

# get path of XCLASS directory
XCLASSRootDir = '/scratch/beck/XCLASS/XCLASS__version_1.4.1/'

# extend sys.path variable
NewPath = XCLASSRootDir + "build_tasks/"
if (not NewPath in sys.path):
    sys.path.append(NewPath)
    
# import XCLASS packages
import task_myXCLASS

original_stdout = sys.stdout
sys.stdout = open('/dev/null', 'w')  # Umleitung in das Null-Gerät

plt.style.use('seaborn')
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
plt.rcParams['font.size'] = 15

if __name__ == "__main__":
    #------------------------------------------------------------------------------------------------------------
    ######################## VALUES DEFINED BY THE USER ########################################################
    #------------------------------------------------------------------------------------------------------------
    #Parameters of the artificial source

    Rho = np.logspace(np.log10(0.001), np.log10(5 * 10**2), 7) # Density at Half Radius in 1/cm^3
    Temperature = np.linspace(50, 400, 7) # Temperature in K at half radius
    Mass_protostar = np.logspace(np.log10(0.1), np.log10(10), 7) # in Solar Masses

    Power_temperature = np.linspace(-3, -0.1, 7)
    Power_density = np.linspace(-3, -0.1, 7)
    
    radius_list = [0.015]
    #radius_list = np.logspace(np.log10(0.005), np.log10(0.2), 10) # radius of the sphere shaped artificial source in pc                                   
    
    dataset = np.empty((0, 4302))  # this is the array for the finished dataset, which is empty at the beginning
    parameterset = np.empty((0, 7))

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------                    
    ######################## LOOP OVER ALL COMBINATIONS OF THE PARAMETERS ##################################################
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Pfad des übergeordneten Verzeichnisses, dessen Unterverzeichnisse gelöscht werden sollen
    directory_path = "/scratch/beck/XCLASS/temp/myXCLASS/"
    molfit_path = '/scratch/beck/XCLASS/XCLASS_Inputfiles/Files/molfit_parallel/'

    # Befehl zum Löschen der Verzeichnisse und deren Inhalte
    command1 = f"rm -r {directory_path}/*"
    command2 = f"rm -r {molfit_path}/*"

    # Befehl im Terminal ausführen
    subprocess.run(command1, shell=True)
    subprocess.run(command2, shell=True)

def process_combination(combination):
    temperature, power_temperature, rho, power_density, mass_protostar, radius = combination

#------------------------------------------------------------------------------------------------------------
######################## VALUES DEFINED BY THE USER ########################################################
#------------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    #Parameters for the spatial cube
    
    Radius = radius/10
    distance = 4400 # distance of the source in parsec
    n_points = 121 # number of pixels of my spatial cube in each direction (x, y, z)
    Restfrequency = 220.74726120 * u.GHz  # set the rest freqeuncy of CH3CN equal to the line centre (from splatalogue)
    # this sets the zero velocity to where the line would be with no relative motion

    #------------------------------------------------------------------------------------------------------------
    #Parameters for the spectral cube

    components = 31 # number of components (in z-direction)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------                    
    ####################### QUANTITIES NEEDED IN THE CALCULATIONS ##################################################
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    G = 6.674 * 10 ** (-11) # Gravitational constant
    M_Sun = 1.98847 * 10 ** 30 # Solar Mass
    pc_in_cm = 3.08567758128 * 10**18 # parsec in cm
    pc_in_m = 30856775814671900 # parsec in m

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
    ############################# MyXCLASS paramters #################################################################################


    # define min. freq. (in MHz)
    FreqMin = 220193.7

    # define max. freq. (in MHz)
    FreqMax = 220767.3

    # define freq. step (in MHz)
    FreqStep = 0.4

    # depending on parameter "Inter_Flag" define beam size (in arcsec)
    # (Inter_Flag = True) or size of telescope (in m) (Inter_Flag = False)
    TelescopeSize = 0.37

    # define beam minor axis length (in arsec)
    BMIN = None

    # define beam major axis length (in arsec)
    BMAJ = None

    # define beam position angle (in degree)
    BPA = None

    # interferrometric data?
    Inter_Flag = True

    # define red shift
    Redshift = None

    # BACKGROUND: describe continuum with tBack and tslope only
    t_back_flag = True

    # BACKGROUND: define background temperature (in K)
    tBack = 0.0

    # BACKGROUND: define temperature slope (dimensionless)
    tslope = 0.0

    # BACKGROUND: define path and name of ASCII file describing continuum as function
    #             of frequency
    BackgroundFileName = ""

    # DUST: define hydrogen column density (in cm^(-2))
    N_H = 1.e24

    # DUST: define spectral index for dust (dimensionless)
    beta_dust = 0.0

    # DUST: define kappa at 1.3 mm (cm^(2) g^(-1))
    kappa_1300 = 0.0

    # DUST: define path and name of ASCII file describing dust opacity as
    #       function of frequency
    DustFileName = ""

    # FREE-FREE: define electronic temperature (in K)
    Te_ff = None

    # FREE-FREE: define emission measure (in pc cm^(-6))
    EM_ff = None

    # SYNCHROTRON: define kappa of energy spectrum of electrons (electrons m^(\u22123) GeV^(-1))
    kappa_sync = None

    # SYNCHROTRON: define magnetic field (in Gauss)
    B_sync = None

    # SYNCHROTRON: energy spectral index (dimensionless)
    p_sync = None

    # SYNCHROTRON: thickness of slab (in AU)
    l_sync = None

    # PHEN-CONT: define phenomenological function which is used to describe
    #            the continuum
    ContPhenFuncID = None

    # PHEN-CONT: define first parameter for phenomenological function
    ContPhenFuncParam1 = None

    # PHEN-CONT: define second parameter for phenomenological function
    ContPhenFuncParam2 = None

    # PHEN-CONT: define third parameter for phenomenological function
    ContPhenFuncParam3 = None

    # PHEN-CONT: define fourth parameter for phenomenological function
    ContPhenFuncParam4 = None

    # PHEN-CONT: define fifth parameter for phenomenological function
    ContPhenFuncParam5 = None

    # use iso ratio file?
    iso_flag = True

    # define path and name of iso ratio file
    IsoTableFileName = "/scratch/beck/XCLASS/XCLASS_Inputfiles/Files/CH3CN_iso.txt"

    # define path and name of file describing Non-LTE parameters
    CollisionFileName = ""

    # define number of pixels in x-direction (used for sub-beam description)
    NumModelPixelXX = 105

    # define number of pixels in y-direction (used for sub-beam description)
    NumModelPixelYY = 105

    # take local-overlap into account or not
    LocalOverlapFlag = False

    # disable sub-beam description
    NoSubBeamFlag = True

    # define path and name of database file
    dbFilename = "/scratch/beck/XCLASS/XCLASS__version_1.4.1/Database/cdms_sqlite.db"

    # define rest freq. (in MHz)
    RestFreq = 220709.01650

    # define v_lsr (in km/s)
    vLSR = 0.0
    
    warnings.filterwarnings("ignore")

    # Generiere eine eindeutige ID für den aktuellen Durchlauf
    run_id = str(uuid.uuid4())

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------                                    
    ################### Making 1D artificial sources #########################################

    N = 100 # number of bins


    radius_array = np.linspace(0, radius, N)  # my 1D artificial source goes from 0 until the radius

    index_Radius = np.argmin(np.abs(radius_array - Radius)) # find one of the indices of x where the little radius lies, where particles do not accelerate anymore
    
    # resulting temperature array and resulting density array for density in 1/cm-3
    temperature_array = temperature * (radius_array/Radius)**power_temperature # for T = a_T * r_half^(alpha_T) 
    temperature_array[:index_Radius] = temperature_array[index_Radius] # setting all values within the little radius equal the to values at its edge
    
    #density_array = rho * (radius_array/Radius)**power_density
    #density_array[:index_Radius] = density_array[index_Radius]

    def get_density(x, y, z):
        ## eingeben in parsec ###
        
        # Berechne den Abstand vom Ursprung für jeden Punkt im Raum
        radius = np.sqrt(x**2 + y**2 + z**2)

        # Berechne die Dichtewerte basierend auf der gegebenen Formel
        density_value = rho * ((radius / Radius)** power_density)
        
        if np.sqrt(x**2 + y**2 + z**2) < Radius:
            density_value = rho

        return density_value # in 1/cm^3


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ######################### Making 3D artificial sources ###############################
        
    # these are the x, y-, and z- coordinates for the cubes that are created in the following steps
    x = np.linspace(-radius, radius, n_points)                                 
    y = np.linspace(-radius, radius, n_points)
    z = np.linspace(-radius, radius, n_points)

    # pixel_widths 
    pixel_width = x[1] - x[0]  # in parsec
    print(pixel_width)
    
    # making a 3-dim array, where every point of the source is represented by a x-, y- and z- coordinate
    X, Y, Z = np.meshgrid(x, y, z)

    # pythagoras - getting my radius-points
    radius_mesh_3D = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)  # now I have a meshgrid which represents a distance

    ## Temperature and Density Cube
    # now my grid where every point describes the radius is interpolated by my 1-dim radius array and therefore by the 1-D temperature- and 1-D density array respectively
    T_grid_3D = np.interp(radius_mesh_3D, radius_array, temperature_array)
    #D_grid_3D = np.interp(radius_mesh_3D, radius_array, density_array)

    ## Velocity-Cube
    radius_vector = np.stack([X, Y, Z], axis=-1)
    r = np.sqrt(np.sum(radius_vector ** 2, axis=-1)) # cube that represents the radius 

    # Calculate the velocity vector for every point in the grid
    velocity_vector = np.sqrt((2* mass_protostar * M_Sun * G) / (r* pc_in_m)**3)[:, :, :, np.newaxis] * radius_vector * pc_in_m

    index_Radius = np.argmin(np.abs(x - Radius)) # find one of the indices of x where the little radius lies, where the particles do not accelerate anymore
    index_center = n_points//2 # the index of the center on the x axis
    Radius_pixel = index_Radius - index_center # the radius of the little radius, where the particles do not accelerate anymore, in pixels              

    x_indices = np.arange(n_points)
    y_indices = np.arange(n_points)
    z_indices = np.arange(n_points)

    X_indices, Y_indices, Z_indices = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij') # A cube that represents all indices of the voxels in the cube

    mask = ((X_indices - index_center)**2 + (Y_indices - index_center)**2 + (Z_indices - index_center)**2) <= Radius_pixel**2 # Here I am making a mask which gives the indices of all voxels in the sphere that lay within the little radius

    velocity_magnitudes = np.linalg.norm(velocity_vector, axis=-1) # Compute the magnitudes of the velocity vectors
    unity_velocity_vector = np.zeros_like(velocity_vector) # Create an array to store the unity vectors
    unity_velocity_vector[mask] = velocity_vector[mask] / velocity_magnitudes[mask, np.newaxis] # Normalize the velocity vectors within the sphere and store them in the unity_velocity_vector array

    V_vector = velocity_vector[index_Radius, index_center, index_center] # Get the veclocity vector at the little radius on the line of sight
    V = np.abs(V_vector[1]) # This is the velocity that the particles have at the little radius "Radius"

    unity_velocity_vector[mask] = V * unity_velocity_vector[mask] # Multiplying the velocity with the unity vectors in the sphere 
    velocity_vector[mask] = unity_velocity_vector[mask] # Transferring these into the velocity_vector-Cube

    V_grid_3D = velocity_vector[:, :, :, 1] #getting the z components of the velocity-vectors 
    V_grid_3D[index_center, :, :] = 0 # particles should definitely be 0 here
    V_grid_3D[index_center, index_center, index_center] = np.nan # in the center the velocity should be a nan-value

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ######################### Creating the Components ###############################

    distance_components_pc = pixel_width * ((n_points//components)+1) # distance between two components in parsec
    
    # I want less components than pixels along x, so I'm taking maps along the line of sight in a certain step
    T_grid_3D = T_grid_3D[::(n_points//(components))+1, :, :] 
    #D_grid_3D = D_grid_3D[::(n_points//(components))+1, :, :]
    #n_grid_3D = D_grid_3D * distance_components_cm
    
    def columndensity(Func, x_min, x_max, nx, y_min, y_max, ny, z_min, z_max, nz):

        ## integrates function Func(x,y,z) in the cuboid [ax,bx] * [ay,by] * [az,bz] using the trapezoidal rule with (nx * ny * nz) integration points
        ## taken from https://books.google.de/books?id=oCVZBAAAQBAJ&pg=PA390&lpg=PA390&dq=trapezoidal+rule+in+3d+python&source=bl&ots
        ##              =qDxRaL-fmt&sig=KbSEJ_tTzFgrvv_1UpYSZQV9h3E&hl=en&sa=X&ved=0ahUKEwj8ktLp9MbUAhVQalAKHa7_AUAQ6AEIYDAJ#v=onepage&q
        ##              =trapezoidal%20rule%20in%203d%20python&f=false

        """
        input parameters:
        -----------------
            - Func:                     function which is integrated
            - x_min:                    lower limit along x-axis
            - x_max:                    upper limit along x-axis
            - nx:                       grid numbers along x-axis
            - y_min:                    lower limit along y-axis
            - y_max:                    upper limit along y-axis
            - ny:                       grid numbers along y-axis
            - z_min:                    lower limit along z-axis
            - z_max:                    upper limit along z-axis
            - nz:                       grid numbers along z-axis

        output parameters:
        -----------------
            - s:                        computed integral
        """

        ## define h for each direction
        hx = (x_max - x_min) / (nx - 1)
        hy = (y_max - y_min) / (ny - 1)
        hz = (z_max - z_min) / (nz - 1)

        S = np.array([])    
        
        for p, q in itertools.product(range(3), range(components)):
            s = 0.0
            for i in range(0, nx):
                x = x_min + i * hx # in parsec
                wx = (hx if i * (i + 1 - nx) else 0.5 * hx) # in parsec
                sx = 0.0
                for j in range(0, ny):
                    y = y_min + j * hy # in parsec
                    wy = (hy if j * (j + 1 - ny) else 0.5 * hy) # in parsec
                    sy = 0.0                
                    for k in range(0, nz):
                        z = z_min + k * hz # in parsec
                        wz = (hz if k * (k + 1 - nz) else 0.5 * hz) # in parsec
                        sy += wz[q] * pc_in_cm * Func(x, y[p], z[q])
                    sx += wy[p] * pc_in_cm * sy
                s += wx * pc_in_cm * sx        
            S = np.append(S, s)
        S = np.reshape(S, (3, 31))
        S = S/((pixel_width * pc_in_cm)**2)
        return S

    V_grid_3D = V_grid_3D[::(n_points//(components))+1, :, :]
    V_grid_3D[components//2, index_center, index_center] = np.nan 
    V_grid_3D = V_grid_3D/1000        

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    ############################### Creating the molfit file for myXCLASS ##################################################
    # Set flags to keep track of modifications for ParameterMapDir and MolfitsFileName
    
    MolfitsFileName = "/scratch/beck/XCLASS/XCLASS_Inputfiles/Files/molfit_parallel/CH3CN_myXCLASS_%s.molfit" % run_id
        
    Spectrum_array = np.array([[]])
    list_i = np.array([0, n_points // 4, n_points//2 - Radius_pixel])

    pix_points = list_i * pixel_width # in parsec

    xKart = radius/2 # in parsec
    yKart = radius - pix_points # in parsec
    zKart = x[::(n_points//(components))+1] # in parsec


    ## define corner coordinates of current cell
    xmin = xKart - (pixel_width * 0.5) # in parsec
    xmax = xKart + (pixel_width * 0.5) # in parsec
    nx = 10                                                                        
    ymin = yKart - (pixel_width * 0.5) # in parsec
    ymax = yKart + (pixel_width * 0.5) # in parsec
    ny = 10                                                                            
    zmin = zKart - (distance_components_pc * 0.5) # in parsec
    zmax = zKart + (distance_components_pc * 0.5) # in parsec
    nz = 10                                                                            

    n_values = columndensity(get_density, xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz)

    for i, j in zip(list_i, np.arange(3)):
        
        line_to_repeat = "n   0.02     2.0        0.371556375883794       y   10.000  1000.00  {}      y  5.000e+8   5.000e+20   {}      y   2.000   30.00  6.00       y   -10.000 10.000  {}    {}"

        file_contents = '''% Number of molecules   1
        %
        % schema:
        %
        % name of molecule              number of components
        % fit:  low:  up:  source size [arcsec]:    fit:  low:  up:  T_rot [K]:    fit:  low:  up:  N_tot [cm-2]:    fit:  low:  up:  velocity width [km/s]: fit:  low:  up:  velocity offset [km/s]:
        %
        CH3CN;v=0;              {}

        '''

        file_contents = file_contents.format(components)
    

        for values in zip(T_grid_3D[:, n_points//2, i], map("{:e}".format, n_values[j, :]), V_grid_3D[::-1, n_points//2, i], range(components, 0, -1)):
            line = line_to_repeat.format(*values)
            file_contents += line + '\n'
            
        with open(MolfitsFileName, "w") as file:
            file.write(file_contents)
            
        with open(MolfitsFileName, "r") as file:
            file_contentsfile_contents = file.read()
            
        #sys.stdout = open('nul', 'w')

########################### call myXCLASS function ##################################################################
        modeldata, log, TransEnergies, IntOpt, JobDir = task_myXCLASS.myXCLASS(
                                                    FreqMin, FreqMax, FreqStep,
                                                    TelescopeSize, BMIN, BMAJ,
                                                    BPA, Inter_Flag, Redshift,
                                                    t_back_flag, tBack, tslope,
                                                    BackgroundFileName,
                                                    N_H, beta_dust, kappa_1300,
                                                    DustFileName, Te_ff, EM_ff,
                                                    kappa_sync, B_sync, p_sync,
                                                    l_sync, ContPhenFuncID,
                                                    ContPhenFuncParam1,
                                                    ContPhenFuncParam2,
                                                    ContPhenFuncParam3,
                                                    ContPhenFuncParam4,
                                                    ContPhenFuncParam5,
                                                    MolfitsFileName, iso_flag,
                                                    IsoTableFileName,
                                                    CollisionFileName,
                                                    NumModelPixelXX,
                                                    NumModelPixelYY,
                                                    LocalOverlapFlag,
                                                    NoSubBeamFlag,
                                                    dbFilename,
                                                    RestFreq, vLSR)
        
        Spectrum_array = np.append(Spectrum_array, modeldata[:, 2])
        parameter_array = np.array([temperature, power_temperature, rho, power_density, mass_protostar, radius, Radius])    
        Frequency_array = modeldata[:, 0]
        
    return Spectrum_array, parameter_array, Frequency_array

if __name__ == "__main__":
    # Erzeugen aller Kombinationen der Parameter
    combinations = list(itertools.product(Temperature, Power_temperature, Rho, Power_density, Mass_protostar, radius_list))

    # Anzahl der Prozesse festlegen (z.B. Anzahl der verfügbaren CPU-Kerne)
    num_processes = 60

    results = []
    with Pool(num_processes) as pool:
        try:
            for result in tqdm(pool.imap_unordered(process_combination, combinations), total=len(combinations)):
                results.append(result)
        except TypeError:
            # Fehler ignorieren und mit dem Code fortfahren
            pass
        pool.close()
        pool.join()


    for Spectrum_array, parameter_array, Frequency_array in results:
        dataset = np.vstack((dataset, Spectrum_array))
        parameterset = np.vstack((parameterset, parameter_array))
        Frequency_array = Frequency_array

    np.savetxt("/home/beck/dataset/dataset.txt", dataset)
    np.savetxt("/home/beck/dataset/parameterset.txt", parameterset)

    sys.stdout = original_stdout

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------           
        ############################## Plotting myXCLASS spectra ###########################################

    Spectrum_array = dataset[0].reshape(3, int(np.shape(dataset)[1]/3))

    plt.style.use('seaborn')

    plt.close('all')
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Spectra of pixels along the radiative cut', fontsize=25)
    grid = Grid(fig, rect=122, nrows_ncols=(3, 1), axes_pad=0.25, label_mode='L')
    list_location = ['pixel at the edge', 'pixel at half radius', 'pixel at the center']

    for k, j, ax in zip([0,1,2], list_location, grid):
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Temperature [K]')
        ax.set_title(j, fontsize=14)
        ax.plot(Frequency_array, Spectrum_array[k, :], 'blue', linewidth = 1)
        ax.set_ylim(0, np.max(Spectrum_array)+10)

    text_box_1 = r'''
    temperature, density at {} pc:

    $T = {} $K,    $\rho = {} \frac{{1}}{{cm^3}}$
    '''.format(parameterset[0, 5], parameterset[0, 0], parameterset[0, 2])

    text_box_ax_1 = fig.add_subplot(111, frame_on=False)
    text_box_ax_1.axis('off')
    text_box_ax_1.text(0, 0.6, text_box_1, size=10,
                        bbox=dict(facecolor='floralwhite', edgecolor='black', boxstyle='round'),
                        fontsize=16, ha='left')

    text_box_2 = r'''
    radius $r = {}$ pc
    mass $M = {} \, M_{{\odot}}$
    '''.format(parameterset[0, 4], parameterset[0, 6])

    text_box_ax_2 = fig.add_subplot(111, frame_on=False)
    text_box_ax_2.axis('off')
    text_box_ax_2.text(0, 0.4, text_box_2, size=10,
                        bbox=dict(facecolor='floralwhite', edgecolor='black', boxstyle='round'),
                        fontsize=16, ha='left')

    text_box_3 = r'''
    power law exponents:

    $\alpha_{{T}} = {}$,   $\alpha_{{\rho}} = {}$ 
    with $T \propto r^{{\, \alpha_{{T}}}}$,   $\rho \propto r^{{\, \alpha_{{\rho}}}}$ 
    '''.format(parameterset[0, 1], parameterset[0, 3])

    text_box_ax_3 = fig.add_subplot(111, frame_on=False)
    text_box_ax_3.axis('off')
    text_box_ax_3.text(0, 0.1, text_box_3, size=10,
                        bbox=dict(facecolor='floralwhite', edgecolor='black', boxstyle='round'),
                        fontsize=16, ha='left')

    plt.tight_layout()
    plt.savefig('/home/beck/10.png')  # Save the image as 'my_plot.png'
    
    # Befehl zum Löschen der Verzeichnisse und deren Inhalte
    command1 = f"rm -r {directory_path}/*"
    command2 = f"rm -r {molfit_path}/*"

    # Befehl im Terminal ausführen
    subprocess.run(command1, shell=True)
    subprocess.run(command2, shell=True)