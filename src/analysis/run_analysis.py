import utils.paths as pth
from analysis.image_analyzer import ImageAnalyzer
from utils.logging import get_logger

logger = get_logger( __name__ )

def analyze():
    for subdir in pth.SUBDIRS:
        analyzer = ImageAnalyzer( subdir, export_images=[ 'gaus_model', 'gaus_resid' ], catalog_format='fits' )
        analyzer.analyze_all_FITS_in_input()

if __name__ == '__main__':
    analyze()
