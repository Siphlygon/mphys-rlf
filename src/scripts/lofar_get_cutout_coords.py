import utils.paths
import h5py
import utils.logging
import requests
import logging
import os
import math

logger = utils.logging.get_logger( "lofar get cutout coords", logging.DEBUG )

def get_cutout(outfile,pos,size=2,low=False,dr3=False,auth=None):
    '''Get a cutout at position pos with size size arcmin. If low is
    True, get the 20-arcsec cutout, else get the 6-arcsec one. If dr3
    is true, try to access the DR3 data instead. Save to outfile.

    '''
    base='dr3' if dr3 else 'dr2'
    url='https://lofar-surveys.org/'
    if low:
        page=base+'-low-cutout.fits'
    else:
        page=base+'-cutout.fits'
    
    logger.debug( f'Trying {url+page}?pos={pos}&size={size}' )
    r=requests.get(url+page,params={'pos':pos,'size':size},auth=auth,stream=True)
    logger.debug( f'received response code {r.status_code} and content type {r.headers["content-type"]}' )
    if r.status_code!=200:
        raise RuntimeError('Status code %i returned' % r.status_code)
    if r.headers['content-type']!='application/fits':
        raise RuntimeError('Server did not return FITS file, probably no coverage of this area')

    with open(outfile,'wb') as o:
        o.write(r.content)
        r.close()



if __name__ == '__main__':
    #image_num = sys.argv[ 1 ]
    image_num = 0

    #dataset_rfa = RecursiveFileAnalyzer( utils.paths.FITS_PARENT / utils.paths.DATASET_SUBDIR )
    #image_fits_file = dataset_rfa.for_each( lambda path : path.name if path.name == f"image{image_num}.fits" else None, 'fits' )
    with h5py.File( str( utils.paths.LOFAR_DATA_PATH ), 'r' ) as h5:
        image_preprocessed = h5[ 'images' ][ image_num ][ :, : ]
        info_array = h5[ 'catalog' ][ 'block1_values' ][ : ][ image_num ]

    RA = info_array[ 0 ]
    DEC = info_array[ 1 ]
    logger.info( f"RA={RA} DEC={DEC}" )
    get_cutout( f"dr2_cutouts/cutout{image_num}.fits", f"{RA} {DEC}", size=2 )

    lower_bound = int( math.floor( image_num / 10000 ) * 10000 )
    upper_bound = int( math.ceil( ( image_num + 1 ) / 10000 ) * 10000 ) - 1
    os.system( f'python src/scripts/fits_viewer.py dr2_cutouts/cutout{image_num}.fits fits_images/dataset/{lower_bound}-{upper_bound}/image{image_num}.fits' )

if False:
    with h5py.File( str( utils.paths.LOFAR_DATA_PATH ), 'r' ) as h5:
        mosaic_ids = h5[ 'catalog' ][ 'axis1' ][ : ]
    print( mosaic_ids )
    catalog = pd.DataFrame( mosaic_ids, columns=[ 'Mosaic_ID' ] )
    download_mosaics( catalog=catalog )
