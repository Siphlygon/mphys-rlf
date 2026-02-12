import configparser
import utils.paths as pth
import numpy as np
import astropy.cosmology
import astropy.units as u
from completeness.img_data_arrays import ImageDataArrays
import matplotlib.pyplot as plt
import matplotlib as mpl

hsv_cmap = mpl.colors.Colormap( 'hsv' )

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read( pth.PROGRAM_CONFIG )
    lu_config = config[ 'loguniform_distribution' ]

    generated_subdir = lu_config[ 'generated_subdir' ]
    dataset_subdir = lu_config[ 'vm_dataset_subdir' ]
    h = float( lu_config[ 'h' ] )
    Tcmb0 = float( lu_config[ 'Tcmb0' ] )
    Om0 = float( lu_config[ 'Om0' ] )
    dz = float( lu_config[ 'dz' ] )
    lum_bins_count = int( lu_config[ 'lum_bins' ] )
    n_pts = int( lu_config[ 'n_pts' ] )

    for subdir in [ dataset_subdir, generated_subdir ]:
        #assign each AGN a random redshift
        data = ImageDataArrays( subdir )
        redshifts = np.random.normal( 0.5, 0.3, len( data.images ) )

        #use the redshift to calculate luminosity distance and luminosity
        cosmo = astropy.cosmology.FlatLambdaCDM( h * u.km / u.s / u.Mpc, Tcmb0=Tcmb0 * u.K, Om0=Om0 )
        luminosity_distances = cosmo.luminosity_distance( redshifts ).to( u.Mpc ).value
        luminosities = data.model_fluxes * ( 4 * np.pi * luminosity_distances**2 )

        #select a redshift-luminosity bin, use monte-carlo to populate the bin,
        # work backwards to find fluxes and weight by completeness function to calculate
        # integral as in Page & Carrera 2000.
        # Also calculate the number of sources in the bin while we're at it
        z_bins = np.arange( 0, 1, dz )
        l_bins = np.linspace( np.min( luminosities ), np.max( luminosities ), lum_bins_count )
        phi_est = np.zeros( (len(z_bins), lum_bins_count) )

        # 
        if ( pth.NP_ARRAY_PARENT / subdir / 'rlf.npy' ).exists():
            phi_est = np.load( pth.NP_ARRAY_PARENT / subdir / 'rlf.npy' )
        else:
            #iteration so we can index phi_est by phi_est[ i_z, i_l ]
            for i_z in range( len( z_bins ) ):
                for i_l in range( lum_bins_count ):
                    z = z_bins[ i_z ]
                    l = l_bins[ i_l ]

                    z_max = z + dz
                    l_max = l + np.max( luminosities ) / lum_bins_count

                    v, v_max = cosmo.comoving_volume( [ z, z_max ] )

                    #we need to generate random comoving volumes and calculate the respective redshifts
                    # this probably should be done inside each bin so we can reliably use np.interp
                    random_volumes = np.random.uniform( v.value, v_max.value, n_pts )
                    redshift_grid = np.logspace( z, z_max, 50 )
                    volume_grid = cosmo.comoving_volume( redshift_grid )
                    random_redshifts = np.interp( random_volumes, volume_grid.value, redshift_grid )

                    #then generate random luminosities -> fluxes
                    # flux values here are in Jy, luminosities in Jy * Mpc**2
                    random_luminosities = np.random.uniform( l, l_max, n_pts )
                    random_luminosity_distances = cosmo.luminosity_distance( random_redshifts ).to( u.Mpc ).value
                    random_fluxes = random_luminosities / ( 4 * np.pi * random_luminosity_distances**2 )

                    #weight each point by Completeness[ flux ] and add to total monte-carlo integral
                    # for now, placeholder, assume flux cutoff at 1 mJy
                    bin_integral = np.sum( random_fluxes > 1e-3 ) / n_pts

                    #now calculate the number of 'real' sources in this bin
                    redshift_mask = ( redshifts >= z ) & ( redshifts < z_max )
                    luminosity_mask = ( luminosities >= l ) & ( luminosities < l_max )
                    N = np.size( data.model_fluxes[ redshift_mask & luminosity_mask ] )

                    #now we have phi_est as given by Page & Carrera 2000
                    phi_est[ i_z, i_l ] = N / bin_integral

            np.save( pth.NP_ARRAY_PARENT / subdir / 'rlf.npy', phi_est )

        #plot the resulting graph
        for i_z in range( phi_est.shape[ 0 ] ):
            specific_phi_est = phi_est[ i_z ]
            plt.plot( l_bins, specific_phi_est, color=hsv_cmap( i_z / phi_est.shape[ 0 ] ), label=f'z={z_bins[ i_z ]}' )
        plt.legend()
        plt.savefig( f'{subdir}_rlf.png' )
