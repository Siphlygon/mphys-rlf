import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math
import argparse
import utils.recursive_file_analyzer as rfa

class FitsViewer:
    """
    A utility class for viewing images in FITS files

    Parameters
    ----------
    *files: list[Path]
        The fits file(s) to read
    """

    def __init__( self, files: tuple[ Path ] ):
        self.files = files
        self.__data_cache = None
        self.__sorting = -1

    NO_SORTING = 0
    SORT_BY_FLUX_SCALED = 1

    def show_image_grid( self,
                         rows: int = -1,
                         resolution: int = 1080,
                         aspect: float = 1,
                         outfile: str | None = None,
                         left: float = 0.05,
                         right: float = 0.95,
                         bottom: float = 0.1,
                         top: float = 0.95,
                         wspace: float = 0.5,
                         hspace: float = 0.5,
                         upper_bound: float | None = None,
                         no_titles: bool = False,
                         no_ticks: bool = False,
                         sorting: int = 0
                          ):
        """
        Show a plot image for the data, and return this object for syntax sugar

        Parameters
        ----------
        rows : int = -1
            Number of rows to display the fits images using, or -1 to determine automatically (square-like)
        resolution : int = 1080
            Vertical pixel size of the full figure (independent of actual pixel size of the image)
        aspect : float = 1
            Aspect ratio for the figure, in terms of width/height.
        outfile : str | None = None
            Where to write the output figure to, or none to not write one
        left : float = 0.05
            Gridspec left parameter
        right : float = 0.95
            Gridspec right parameter
        bottom : float = 0.1
            Gridspec bottom parameter
        top : float = 0.95
            Gridspec top parameter
        wspace : float = 0.5
            Gridspec wspace parameter
        hspace : float = 0.5
            Gridspec hspace parameter
        upper_bound : float | None = None
            The float value to set as the upper bound for the images, to have a uniform scale. None gives each image its own scale.
        no_titles : bool = False
            Whether or not to hide the titles, to declutter large grids. Default false.
        no_ticks : bool = False
            Whether or not to hide the ticks, to declutter large grids. Default false.

        """
        if self.files is None or len( self.files ) == 0:
            raise RuntimeError( "Cannot call show_image with no data" )
    
        # Check if we have a cached data array we can use, and if we don't get the data from self.files and set cache variables
        # The cached data array should be valid unless we have a different sorting
        if self.__data_cache is not None and self.__sorting == sorting:
            data = self.__data_cache
        else:
            if sorting == FitsViewer.NO_SORTING:
                files = self.files
            elif sorting == FitsViewer.SORT_BY_FLUX_SCALED:
                files = sorted( self.files, key=lambda file : rfa.get_fits_primaryhdu_header( file, 'FXSCLD' ) )

            data = [ rfa.get_fits_primaryhdu_data( files[ i ] ) for i in range( len( files ) ) ] # Order is important here, use index iter just in case
            self.__data_cache = data
            self.__sorting = sorting

        titles = [ None if no_titles else files[ i ].name for i in range( len( self.files ) ) ] # Order is also important here

        fig = plt.figure( figsize=(aspect*resolution/100, resolution/100) )

        if rows == -1:
            rows = math.ceil( math.sqrt( len( titles ) ) / aspect )
        num_cols = math.ceil( len( titles ) / rows )
        gs = fig.add_gridspec( rows, num_cols,
                               left=left, right=right, bottom=bottom, top=top,
                               wspace=wspace, hspace=hspace )
        axes: list[ plt.Axes ] = []
        for i in range( len( titles ) ):
            col = i // rows
            row = i % rows
            axes.append( fig.add_subplot( gs[ row, col ] ) )
            if titles[ i ] is not None:
                axes[ -1 ].set_title( titles[ i ] )
            img = axes[ -1 ].imshow( data[ i ] )
            if upper_bound:
                img.set_clim( 0, upper_bound )

            if no_ticks:
                axes[ -1 ].xaxis.set_visible( False )
                axes[ -1 ].yaxis.set_visible( False )

        if outfile is not None:
            plt.savefig( outfile )
        plt.show()
        return self

if __name__ == "__main__":
    # Display any fits files passed as arguments
    parser = argparse.ArgumentParser( prog='python fits_viewer.py', 
                                      description='A program to visualize FITS images passed as arguments' )
    parser.add_argument( "--rows", help="How many rows to display the fits images in, default -1 automatically makes the image square-like", type=int, default=-1 )
    parser.add_argument( "--resolution", help="Vertical pixel resolution to display the image grid at, default 1080", type=int, default=1080 )
    parser.add_argument( "--aspect", help="Aspect ratio to display the image grid at, default 1", type=float, default=1 )
    parser.add_argument( "-o", "--outfile", help="Where to write the output file to. By default does not write figure to a file.", type=str, default=None )
    parser.add_argument( "--left", help="Gridspec left parameter default 0.05", type=float, default=0.05 )
    parser.add_argument( "--right", help="Gridspec right parameter default 0.95", type=float, default=0.95 )
    parser.add_argument( "--top", help="Gridspec top parameter default 0.95", type=float, default=0.95 )
    parser.add_argument( "--bottom", help="Gridspec bottom parameter default 0.1", type=float, default=0.1 )
    parser.add_argument( "--wspace", help="Gridspec wspace parameter default 0.5", type=float, default=0.5 )
    parser.add_argument( "--hspace", help="Gridspec hspace parameter default 0.5", type=float, default=0.5 )
    parser.add_argument( "--upper-bound", help="Specify upper bound on images, by default no bound", type=float, default=None )
    parser.add_argument( "--no-titles", help="Use this flag to not display titles on the images, e.g. for cluttered grids", action='store_true', default=False )
    parser.add_argument( "--no-ticks", help="Use this flag to not display axis ticks on the images, e.g. for cluttered grids", action='store_true', default=False )
    parser.add_argument( "--sorting", help="0 does no sorting, 1 sorts by peak flux scaled, 2 sorts by unscaled peak flux. Note that these should be identical up to a minus sign.", type=int, default=0 )
    parser.add_argument( "FILES", help="Any number of files, or directories to search through, to be read and displayed in a grid", nargs='*' )
    args = parser.parse_args()

    if len( args.FILES ) > 0:
        paths: list[ Path ] = []
        for element in args.FILES:
            element_path = Path( element )
            if element_path.is_dir():
                for sub_elements in element_path.rglob( "*" ):
                    paths.append( sub_elements )
            else:
                paths.append( element_path )

        kwargs = vars( args )
        kwargs.pop( 'FILES' ) #Already interpreted, shouldn't be passed to show_image_grid
        viewer = FitsViewer( paths ).show_image_grid( **kwargs )

