import argparse
import textwrap

import configargparse

# from falcon import __version__


class NewlineTextHelpFormatter(argparse.HelpFormatter):

    def _fill_text(self, text, width, indent):
        return '\n'.join(
            textwrap.fill(line, width, initial_indent=indent,
                          subsequent_indent=indent,
                          replace_whitespace=False).strip()
            for line in text.splitlines(keepends=True))


class Config:
    """
    Commandline and file-based configuration.

    Configuration settings can be specified in a config.ini file (by default in
    the working directory), or as command-line arguments.
    """

    def __init__(self) -> None:
        """
        Initialize the configuration settings and provide sensible default
        values if possible.
        """

        self._parser = configargparse.ArgParser(
            # description=f'falcon: Fast spectrum clustering using nearest '
            #             f'neighbor searching\n'
            #             f'==============================================='
            #             f'==================\n\n'
            #             f'falcon version {__version__}\n\n'
            #             f'Official code website: '
            #             f'https://github.com/bittremieux/falcon\n\n',
            default_config_files=['config.ini'],
            args_for_setting_config_path=['-c', '--config'],
            formatter_class=NewlineTextHelpFormatter)

        # IO
        self._parser.add_argument(
            'input_filenames', #nargs='+',
            help='Input peak files (supported formats: .mzML, .mzXML, .MGF).')
        self._parser.add_argument(
            'annotations_file', help='Annotations file name.')
        # self._parser.add_argument(
        #     'output_filename', help='Output file name.')
        # self._parser.add_argument(
        #     '--work_dir', default=None,
        #     help='Working directory (default: temporary directory).')
        # self._parser.add_argument(
        #     '--overwrite', action='store_true',
        #     help="Overwrite existing results (default: don't overwrite).")
        # self._parser.add_argument(
        #     '--export_representatives', action='store_true',
        #     help='Export cluster representatives to an MGF file '
        #          '(default: no export).')
        self._parser.add_argument(
            '--dist_matrix_file', type=str, default=None,
            help='Precomputed distance matrix file in .npz format')
        self._parser.add_argument(
            '--matches_matrix_file', type=str, default=None,
            help='Precomputed matches matrix file in .npz format')
        self._parser.add_argument(
            '--export_dist_matrix', action='store_true',
            help='Export distance matrix as .npz file (default: no export)')

        # CLUSTERING
        self._parser.add_argument(
            '--precursor_tol', nargs=2, default=[20, 'ppm'],
            help='Precursor tolerance mass and mode (default: 20 ppm). '
                 'Mode should be either "ppm" or "Da".')
        # self._parser.add_argument(
        #     '--rt_tol', type=float, default=None,
        #     help='Retention time tolerance (default: no retention time '
        #          'filtering).')
        self._parser.add_argument(
            '--fragment_tol', type=float, default=0.05,
            help='Fragment mass tolerance in m/z (default: %(default)s m/z).')

        # self._parser.add_argument(
        #     '--mz_interval', type=int, default=1,
        #     help='Precursor m/z interval (centered around x.5 Da) to process '
        #          'spectra simultaneously (default: %(default)s m/z).')
        # self._parser.add_argument(
        #     '--low_dim', default=400, type=int,
        #     help='Low-dimensional vector length (default: %(default)s).')
        # self._parser.add_argument(
        #     '--n_neighbors', default=64, type=int,
        #     help='Number of neighbors to include in the pairwise distance '
        #          'matrix for each spectrum (default: %(default)s).')
        # self._parser.add_argument(
        #     '--n_neighbors_ann', default=128, type=int,
        #     help='Number of neighbors to retrieve from the nearest neighbor '
        #          'indexes prior to precursor tolerance filtering '
        #          '(default: %(default)s).')
        # self._parser.add_argument(
        #     '--batch_size', default=2**16, type=int,
        #     help='Number of spectra to process simultaneously '
        #          '(default: %(default)s).')
        # self._parser.add_argument(
        #     '--n_probe', default=32, type=int,
        #     help='Maximum number of lists in the inverted index to inspect '
        #          'during querying (default: %(default)s).')

        self._parser.add_argument(
            '--cluster_method', default='hierarchical', type=str,
            help='Clustering method to use, "hierarchical" or "DBSCAN" '
            '(default: %(default)s).')
        self._parser.add_argument(
            '--min_cluster_size', default=2, type=int,
            help='Minimum cluster size. In HC: samples in clusters of size < min_cluster_size '
            'will be labeled as noise. In DBSCAN: corresponds to min_samples parameter '
            '(see sklearn docs). (default: %(default)s).')
        self._parser.add_argument(
            '--linkage', default='complete', type=str,
            help='Linkage criterion to use for hierarchical clustering, see sklearn docs '
            '(default: %(default)s).')
        self._parser.add_argument(
            '--max_cluster_dist', default=1, type=float,
            help='Maximum distance above which clusters will not be merged anymore '
            '(default: %(default)s).')
        self._parser.add_argument(
            '--plot_dendrogram', action='store_true',
            help='Plot dendrogram, only for hierarchical clustering (default: no plot).')
        self._parser.add_argument(
            '--eps', type=float, default=0.2,
            help='The eps parameter (modified cosine distance) for DBSCAN clustering '
            '(default: %(default)s). Relevant cosine distance thresholds '
            'are typically between 0.05 and 0.30.')

        # PREPROCESSING
        self._parser.add_argument(
            '--min_peaks', default=5, type=int,
            help='Discard spectra with fewer than this number of peaks '
                 '(default: %(default)s).')
        self._parser.add_argument(
            '--min_mz_range', default=50., type=float,
            help='Discard spectra with a smaller mass range '
                 '(default: %(default)s m/z).')
        self._parser.add_argument(
            '--min_mz', default=51., type=float,
            help='Minimum peak m/z value (inclusive, '
                 'default: %(default)s m/z).')
        self._parser.add_argument(
            '--max_mz', default=1500., type=float,
            help='Maximum peak m/z value (inclusive, '
                 'default: %(default)s m/z).')
        self._parser.add_argument(
            '--remove_precursor_tol', default=1.5, type=float,
            help='Window around the precursor mass to remove peaks '
                 '(default: %(default)s m/z).')
        self._parser.add_argument(
            '--min_intensity', default=0.01, type=float,
            help='Remove peaks with a lower intensity relative to the base '
                 'intensity (default: %(default)s).')
        self._parser.add_argument(
            '--max_peaks_used', default=50, type=int,
            help='Only use the specified most intense peaks in the spectra '
                 '(default: %(default)s).')
        self._parser.add_argument(
            '--scaling', default='off', type=str,
            choices=['off', 'root', 'log', 'rank'],
            help='Peak scaling method used to reduce the influence of very '
                 'intense peaks (default: %(default)s).')

        # MOLECULAR NETWORKING
        self._parser.add_argument(
            '--max_edge_dist', default=0.2, type=float,
            help='Maximum distance between spectra to add edge to the molecular '
            'network (default: %(default)s).')
        self._parser.add_argument(
            '--max_edges', default=None, type=int,
            help='Maximum number of edges for each node in the network. '
            'If None, all edges are shown (default: %(default)s).')
        self._parser.add_argument(
            '--min_matched_peaks', default=0, type=int,
            help='Minimum number of matching peaks between spectra to add edge to '
            'the molecular network (default: %(default)s).')

        # Filled in 'parse', contains the specified settings.
        self._namespace = None

    def parse(self, args_str: str = None) -> None:
        """
        Parse the configuration settings.

        Parameters
        ----------
        args_str : str
            If None, the arguments are taken from sys.argv. Arguments that are
            not explicitly specified are taken from the configuration file.
        """
        self._namespace = vars(self._parser.parse_args(args_str))

        self._namespace['precursor_tol'][0] = \
            float(self._namespace['precursor_tol'][0])

    def __getattr__(self, option):
        if self._namespace is None:
            raise RuntimeError('The configuration has not been initialized')
        return self._namespace[option]

    def __getitem__(self, item):
        return self.__getattr__(item)


config = Config()