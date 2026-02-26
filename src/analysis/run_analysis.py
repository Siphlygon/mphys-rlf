import utils.paths as pth
from analysis.image_analyzer import ImageAnalyzer
from utils.logging import get_logger
import sys
import argparse

logger = get_logger( __name__ )

def analyze( subdirs: list[ str ], fits_input_dir: str | None = None ):
    for subdir in subdirs:
        if fits_input_dir is None:
            fits_input_dir = pth.FITS_PARENT

        analyzer = ImageAnalyzer( subdir, export_images=[ 'gaus_model', 'gaus_resid' ], catalog_format='fits', fits_input_dir=fits_input_dir )
        analyzer.analyze_all_FITS_in_input()

if __name__ == '__main__':
    parser = argparse.ArgumentParser( prog='python run_analysis.py', 
                                      description='A program to analyze fits files with PyBDSF' )
    parser.add_argument( "--input-dir", help="FITS input directory, default utils.paths.FITS_PARENT", type=str, default=None )
    parser.add_argument( "SUBDIRS", help="Any number of subdirectories, or 0 to use utils.paths.SUBDIRS", nargs='*' )
    args = parser.parse_args()

    if len( args.SUBDIRS ) > 0:
        logger.info( f"Analyzing {len( args.SUBDIRS )} custom subdirs" )
        for subdir in args.SUBDIRS:
            logger.info( f"    {subdir}" )
        logger.info( f"fits input directory {args.input_dir}" )
        analyze( args.SUBDIRS, args.input_dir )
    else:
        logger.info( f"Analyzing default subdirectories from utils.paths.SUBDIRS" )
        analyze( pth.SUBDIRS )
