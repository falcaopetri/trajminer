import numpy as np

class TrajectoryData:
    """Trajectory data wrapper.

    Parameters
    ----------
    data : pandas.DataFrame, shape: (n_trajectories * n_points, n_features)
        The trajectory data with a 'tid' column.
    labels : pandas.Series (default=None)
        The corresponding labels of trajectories in ``data``, indexed by 'tid'.
    """

    def __init__(self, df, labels=None):
        if 'tid' not in df.columns:
            raise ValueError("Trajectory dataframe must have a 'tid' column.")

        df = df.sort_values(by='tid').reset_index(drop=True)
        labels = labels.sort_index()

        if labels is not None and not np.array_equal(df['tid'].unique(), labels.index):
            raise ValueError("Trajectory dataframe tids must match with labels's index.")

        self.data = df
        self.labels = labels

    def get_attributes(self):
        """Retrieves the attributes in the dataset.

        Returns
        -------
        attributes : array
            An array of length `n_features`.
        """
        return self.attributes

    def get_tids(self, label=None):
        """Retrieves the trajectory IDs in the dataset.

        Parameters
        ----------
        label : int (default=None)
            If `None`, then retrieves all trajectory IDs. Otherwise, returns
            the IDs corresponding to the given label.

        Returns
        -------
        attributes : array
            An array of length `n_trajectories`.
        """
        if not label or self.labels is None:
            return self.tids

        idxs = self.labelToIdx[label]
        return self.tids[idxs]

    def get_label(self, tid):
        """Retrieves the label for the corresponding tid.

        Parameters
        ----------
        tid : int
            The trajectory ID.

        Returns
        -------
        label : int or str
            The corresponding label.
        """
        return self.labels[self.tidToIdx[tid]]

    def get_labels(self, unique=False):
        """Retrieves the labels of the trajectories in the dataset.

        Parameters
        ----------
        unique : bool (default=False)
            If ``True``, then the set of unique labels is returned. Otherwise,
            an array with the labels of each individual trajectory is returned.

        Returns
        -------
        labels : array
            An array of length `n_trajectories` if `unique=False`, and of
            length `n_labels` otherwise.
        """
        if unique and self.labels is not None:
            return sorted(list(set(self.labels)))

        return self.labels

    def get_trajectory(self, tid):
        """Retrieves a trajectory from the dataset.

        Parameters
        ----------
        tid : int
            The trajectory ID.

        Returns
        -------
        trajectory : array, shape: (n_points, n_features)
            The corresponding trajectory.
        """
        return self.data[self.tidToIdx[tid]]

    def get_trajectories(self, label=None):
        """Retrieves multiple trajectories from the dataset.

        Parameters
        ----------
        label : int (default=None)
            The label of the trajectories to be retrieved. If ``None``, then
            all trajectories are retrieved.

        Returns
        -------
        trajectories : array
            The trajectories of the given label. If `label=None` or if the
            dataset does not contain labels, then all trajectories are
            returned.
        """
        if not label or self.labels is None:
            return self.data

        idxs = self.labelToIdx[label]
        return self.data[idxs]

    def length(self):
        """Returns the number of trajectories in the dataset.

        Returns
        -------
        length : int
            Number of trajectories in the dataset.
        """
        return len(self.tids)

    def merge(self, other, ignore_duplicates=True, inplace=True):
        """Merges this trajectory data with another one. Notice that this
        method only works if the datasets have the same set of attributes.

        Parameters
        ----------
        other : :class:`trajminer.TrajectoryData`
            The dataset to be merged with.
        ignore_duplicates : bool (default=True)
            If `True`, then trajectory IDs in `other` that already exist in
            `self` are ignored. Otherwise, raises an exception when a duplicate
            is found.
        inplace : bool (default=True)
            If `True` modifies the current object, otherwise returns a new
            object.

        Returns
        -------
        dataset : :class:`trajminer.TrajectoryData`
            The merged dataset. If `inplace=True`, then returns the modified
            current object.
        """
        if set(self.attributes) != set(other.attributes):
            raise Exception("Cannot merge datasets with different sets of " +
                            "attributes!")

        n_attributes = self.attributes
        n_tids = self.tids.tolist()
        n_labels = self.labels.tolist()
        n_data = self.data.tolist()

        for tid in other.tids:
            if tid in n_tids:
                if ignore_duplicates:
                    continue
                raise Exception("tid", tid, "already exists in 'self'!")
            n_tids.append(tid)
            n_data.append(other.get_trajectory(tid))

            if n_labels is not None:
                n_labels.append(other.get_label(tid))

        if inplace:
            self._update(n_attributes, n_data, n_tids, n_labels)
            return self

        return TrajectoryData(n_attributes, n_data, n_tids, n_labels)

    def to_file(self, file, file_type='csv', n_jobs=1):
        """Persists the dataset to a file.

        Parameters
        ----------
        file : str
            The output file.
        file_type : str (default='csv')
            The file type. Must be one of `{csv}`.
        n_jobs : int (default=1)
            The number of parallel jobs.
        """
        if file_type == 'csv':
            self._to_csv(file, n_jobs)

    def stats(self, print_stats=False):
        """Computes statistics for the dataset.

        Parameters
        ----------
        print_stats : bool (default=False)
            If `True`, stats are printed.

        Returns
        -------
        stats : dict
            A dictionary containing the dataset statistics.
        """
        if self._stats:
            if print_stats:
                self._print_stats()
            return self._stats

        traj_lengths = np.array([len(x) for x in self.data])
        points = np.concatenate(self.data)

        def count_not_none(arr):
            return sum([1 if x is not None else 0 for x in arr])

        attr_count = np.array([count_not_none(p) for p in points])

        self._stats = {
            'attribute': {
                'count': len(self.attributes),
                'min': attr_count.min(),
                'avg': attr_count.mean(),
                'std': attr_count.std(),
                'max': attr_count.max()
            },
            'point': {
                'count': traj_lengths.sum()
            },
            'trajectory': {
                'count': len(self.data),
                'length': {
                    'min': traj_lengths.min(),
                    'avg': traj_lengths.mean(),
                    'std': traj_lengths.std(),
                    'max': traj_lengths.max()
                }
            }
        }

        if self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            self._stats['label'] = {
                'count': len(unique),
                'min': counts.min(),
                'avg': counts.mean(),
                'std': counts.std(),
                'max': counts.max()
            }

        if print_stats:
            self._print_stats()
        return self._stats

    def _update(self, attributes, data, tids, labels):
        self.tids = np.array(tids)
        self.labels = np.array(labels)
        self.data = np.array(data)
        self.tidToIdx = dict(zip(tids, np.r_[0:len(tids)]))
        self.labelToIdx = TrajectoryData._get_label_to_idx(labels)
        self._stats = None

    def _to_csv(self, file, n_jobs):
        lat_lon = -1
        tids = self.get_tids()

        def build_lines(s):
            lines = []
            for i in range(s.start, s.stop):
                tid = tids[i]
                label = self.get_label(tid)
                traj = self.get_trajectory(tid)

                for p in traj:
                    if lat_lon > -1:
                        p[lat_lon] = str(p[lat_lon][0]) + \
                            ',' + str(p[lat_lon][1])
                    fmt = str(p)[1:-1].replace(', ', ',').replace("'", '')
                    lines.append(str(tid) + ',' + str(label) + ',' + fmt)
            return lines

        with open(file, 'w') as out:
            header = 'tid,label'

            for i, attr in enumerate(self.get_attributes()):
                if attr == 'lat_lon':
                    header += ',lat,lon'
                    lat_lon = i
                else:
                    header += ',' + attr

            out.write(header + '\n')
            func = delayed(build_lines)
            lines = Parallel(n_jobs=n_jobs, verbose=0)(
                func(s) for s in gen_even_slices(len(tids), n_jobs))

            lines = np.concatenate(lines)
            lines = '\n'.join(lines)
            out.write(lines)
            out.close()

    def _print_stats(self):
        print('==========================================================')
        print('                           STATS                          ')
        print('==========================================================')
        print('ATTRIBUTE')
        print('  Count:           ', self._stats['attribute']['count'])
        print('  Min:             ', self._stats['attribute']['min'])
        print('  Max:             ', self._stats['attribute']['max'])
        print('  Avg ± Std:        %.4f ± %.4f' % (
            self._stats['attribute']['avg'], self._stats['attribute']['std']))

        print('\nPOINT')
        print('  Count:           ', self._stats['point']['count'])

        print('\nTRAJECTORY')
        print('  Count:           ', self._stats['trajectory']['count'])
        print('  Min length:      ',
              self._stats['trajectory']['length']['min'])
        print('  Max lenght:      ',
              self._stats['trajectory']['length']['max'])
        print('  Avg length ± Std: %.4f ± %.4f' %
              (self._stats['trajectory']['length']['avg'],
               self._stats['trajectory']['length']['std']))

        if self.labels is not None:
            print('\nLABEL')
            print('  Count:           ', self._stats['label']['count'])
            print('  Min:             ', self._stats['label']['min'])
            print('  Max:             ', self._stats['label']['max'])
            print('  Avg ± Std:        %.4f ± %.4f' % (
                self._stats['label']['avg'], self._stats['label']['std']))
            print('==========================================================')
        else:
            print('==========================================================')

    @staticmethod
    def _get_label_to_idx(labels):
        labelToIdx = None
        if labels is not None:
            labelToIdx = {}
            for i, label in enumerate(labels):
                if label in labelToIdx:
                    labelToIdx[label].append(i)
                else:
                    labelToIdx[label] = [i]

        return labelToIdx
