import pandas as pd
import numpy as np

from ..trajectory_data import TrajectoryData


class TrajectoryLoader(object):
    """Base class for trajectory loaders.
    """

    def load(self):
        """Loads trajectories according to the specific approach.

        Returns
        -------
        data : :class:`trajminer.TrajectoryData`
            A :class:`trajminer.TrajectoryData` containing the loaded dataset.
        """
        pass


class CSVTrajectoryLoader(TrajectoryLoader):
    """A trajectory data loader from a CSV file.

    Parameters
    ----------
    file : str
        The CSV file from which to read the data.
    sep : str (default=',')
        The CSV separator.
    tid_col : str (default='tid')
        The column in the CSV file corresponding to the trajectory IDs.
    label_col : str (default='label')
        The column in the CSV file corresponding to the trajectory labels. If
        `None`, labels are not loaded.
    lat : str (default='lat')
        The column in the CSV file corresponding to the latitude of the
        trajectory points.
    lon : str (default='lon')
        The column in the CSV file corresponding to the longitude of the
        trajectory points.
    drop_col : array-like (default=None)
        List of columns to drop when reading the data from the file.

    Examples
    --------
    >>> from trajminer.utils import CSVTrajectoryLoader
    >>> loader = CSVTrajectoryLoader('my_data.csv')
    >>> dataset = loader.load()
    >>> dataset.get_attributes()
    ['poi', 'day', 'time']
    """

    def __init__(self, file, sep=',', tid_col='tid', label_col='label',
                 lat='lat', lon='lon', drop_col=None, n_jobs=1):
        self.file = file
        self.sep = sep
        self.tid_col = tid_col
        self.label_col = label_col
        self.lat = lat
        self.lon = lon
        self.drop_col = drop_col if drop_col is not None else []

    def load(self):
        df = pd.read_csv(self.file, sep=self.sep, usecols=lambda c: c not in self.drop_col)
        df = df.rename(columns={self.tid_col: 'tid', self.lat: 'lat', self.lon: 'lon'})

        if self.label_col is None:
            # no label_col specified
            return TrajectoryData(df, labels=None)

        real_label_col = self.label_col
        if self.label_col == self.tid_col:
            # update label_col since we renamed tid_col to 'tid'
            real_label_col = 'tid'

        # get real_label_col series indexed by 'tid'
        labels = df.set_index('tid', drop=False)[real_label_col].drop_duplicates()

        if real_label_col != 'tid':
            # real_label_col is not 'tid', so we can drop the real_label_col from data
            df = df.drop(columns=real_label_col)

        return TrajectoryData(df, labels)
