"""Session creation and datasets registration.

The RegistrationClient provides an high-level API for creating experimentation sessions on Alyx
and registering associated datasets.

Summary of methods
------------------
create_new_session - Create a new local session folder and optionally create session record on Alyx
create_sessions - Create sessions and register files for folder containing a given flag file
register_session - Create a session on Alyx from local path and register any ALF datasets present
register_files - Register a list of files to their respective sessions on Alyx
"""
import pathlib
from uuid import UUID
from pathlib import Path, PurePosixPath
import datetime
import logging
import itertools
from collections import defaultdict
from fnmatch import fnmatch
import shutil

import requests.exceptions

from iblutil.io import hashfile
from iblutil.util import Bunch

import one.alf.io as alfio
from one.alf.files import session_path_parts, get_session_path, folder_parts, filename_parts
from one.alf.spec import is_valid
import one.alf.exceptions as alferr
from one.api import ONE
from one.util import ensure_list
from one.webclient import no_cache

_logger = logging.getLogger(__name__)


def get_dataset_type(filename, dtypes):
    """Get the dataset type from a given filename.

    A dataset type is matched one of two ways:

     1. the filename matches the dataset type filename_pattern;
     2. if filename_pattern is empty, the filename object.attribute matches the dataset type name.

    Parameters
    ----------
    filename : str, pathlib.Path
        The filename or filepath.
    dtypes : iterable
        An iterable of dataset type objects with the attributes ('name', 'filename_pattern').

    Returns
    -------
    The matching dataset type object for filename.

    Raises
    ------
    ValueError
        filename doesn't match any of the dataset types
        filename matches multiple dataset types
    """
    dataset_types = []
    if isinstance(filename, str):
        filename = PurePosixPath(filename)
    for dt in dtypes:
        if not dt.filename_pattern.strip():
            # If the filename pattern is null, check whether the filename object.attribute matches
            # the dataset type name.
            if is_valid(filename.name):
                obj_attr = '.'.join(filename_parts(filename.name)[1:3])
            else:  # will match name against filename sans extension
                obj_attr = filename.stem
            if dt.name == obj_attr:
                dataset_types.append(dt)
        # Check whether pattern matches filename
        elif fnmatch(filename.name.lower(), dt.filename_pattern.lower()):
            dataset_types.append(dt)
    n = len(dataset_types)
    if n == 0:
        raise ValueError(f'No dataset type found for filename "{filename.name}"')
    elif n >= 2:
        raise ValueError('Multiple matching dataset types found for filename '
                         f'"{filename.name}": \n{", ".join(map(str, dataset_types))}')
    return dataset_types[0]


class RegistrationClient:
    """
    Object that keeps the ONE instance and provides method to create sessions and register data.
    """
    def __init__(self, one=None):
        self.one = one
        if not one:
            self.one = ONE(cache_rest=None)
        self.dtypes = list(map(Bunch, self.one.alyx.rest('dataset-types', 'list')))
        self.registration_patterns = [
            dt['filename_pattern'] for dt in self.dtypes if dt['filename_pattern']]
        self.file_extensions = [df['file_extension'] for df in
                                self.one.alyx.rest('data-formats', 'list', no_cache=True)]

    def create_sessions(self, root_data_folder, glob_pattern='**/create_me.flag',
                        register_files=False, dry=False):
        """
        Create sessions looking recursively for flag files.

        Parameters
        ----------
        root_data_folder : str, pathlib.Path
            Folder to look for sessions.
        glob_pattern : str
            Register valid sessions that contain this pattern.
        register_files : bool
            If true, register all valid datasets within the session folder.
        dry : bool
            If true returns list of sessions without creating them on Alyx.

        Returns
        -------
        list of pathlib.Paths
            Newly created session paths.
        list of dicts
            Alyx session records.
        """
        flag_files = list(Path(root_data_folder).glob(glob_pattern))
        records = []
        for flag_file in flag_files:
            if dry:
                records.append(print(flag_file))
                continue
            _logger.info('creating session for ' + str(flag_file.parent))
            # providing a false flag stops the registration after session creation
            session_info, _ = self.register_session(flag_file.parent, file_list=register_files)
            records.append(session_info)
            flag_file.unlink()
        return [ff.parent for ff in flag_files], records

    def create_new_session(self, subject, session_root=None, date=None, register=True, **kwargs):
        """Create a new local session folder and optionally create session record on Alyx.

        Parameters
        ----------
        subject : str
            The subject name.  Must exist on Alyx.
        session_root : str, pathlib.Path
            The root folder in which to create the subject/date/number folder.  Defaults to ONE
            cache directory.
        date : datetime.datetime, datetime.date, str
            An optional date for the session.  If None the current time is used.
        register : bool
            If true, create session record on Alyx database.
        **kwargs
            Optional arguments for RegistrationClient.register_session.

        Returns
        -------
        pathlib.Path
            New local session path.
        uuid.UUID
            The experiment UUID if register is True.

        Examples
        --------
        Create a local session only

        >>> session_path, _ = RegistrationClient().create_new_session('Ian', register=False)

        Register a session on Alyx in a specific location

        >>> session_path, eid = RegistrationClient().create_new_session('Sy', '/data/lab/Subjects')

        Create a session for a given date

        >>> session_path, eid = RegistrationClient().create_new_session('Ian', date='2020-01-01')
        """
        assert not self.one.offline, 'ONE must be in online mode'
        date = self.ensure_ISO8601(date)  # Format, validate
        # Ensure subject exists on Alyx
        self.assert_exists(subject, 'subjects')
        session_root = Path(session_root or self.one.alyx.cache_dir) / subject / date[:10]
        session_path = session_root / alfio.next_num_folder(session_root)
        session_path.mkdir(exist_ok=True, parents=True)  # Ensure folder exists on disk
        if register:
            session_info, _ = self.register_session(session_path, **kwargs)
            eid = UUID(session_info['url'][-36:])
        else:
            eid = None
        return session_path, eid

    def find_files(self, session_path):
        """
        Returns an generator of file names that match one of the dataset type patterns in Alyx

        Parameters
        ----------
        session_path : str, pathlib.Path
            The session path to search

        Yields
        -------
        pathlib.Path
            File paths that match the dataset type patterns in Alyx
        """
        session_path = Path(session_path)
        for p in session_path.rglob('*.*.*'):
            if p.is_file() and any(p.name.endswith(ext) for ext in self.file_extensions):
                try:
                    get_dataset_type(p, self.dtypes)
                    yield p
                except ValueError as ex:
                    _logger.debug('%s', ex.args[0])

    def assert_exists(self, member, endpoint):
        """Raise an error if a given member doesn't exist on Alyx database.

        Parameters
        ----------
        member : str, uuid.UUID, list
            The member ID(s) to verify
        endpoint: str
            The endpoint at which to look it up

        Examples
        --------
        >>> client.assert_exists('ALK_036', 'subjects')
        >>> client.assert_exists('user_45', 'users')
        >>> client.assert_exists('local_server', 'repositories')

        Raises
        -------
        one.alf.exceptions.AlyxSubjectNotFound
            Subject does not exist on Alyx
        one.alf.exceptions.ALFError
            Member does not exist on Alyx
        requests.exceptions.HTTPError
            Failed to connect to Alyx database or endpoint not found

        Returns
        -------
        dict, list of dict
            The endpoint data if member exists.
        """
        if isinstance(member, (str, UUID)):
            try:
                return self.one.alyx.rest(endpoint, 'read', id=str(member), no_cache=True)
            except requests.exceptions.HTTPError as ex:
                if ex.response.status_code != 404:
                    raise ex
                elif endpoint == 'subjects':
                    raise alferr.AlyxSubjectNotFound(member)
                else:
                    raise alferr.ALFError(f'Member "{member}" doesn\'t exist in {endpoint}')
        else:
            return [self.assert_exists(x, endpoint) for x in member]

    @staticmethod
    def ensure_ISO8601(date) -> str:
        """Ensure provided date is ISO 8601 compliant

        Parameters
        ----------
        date : str, None, datetime.date, datetime.datetime
            An optional date to convert to ISO string.  If None, the current datetime is used.

        Returns
        -------
        str
            The datetime as an ISO 8601 string
        """
        date = date or datetime.datetime.now()  # If None get current time
        if isinstance(date, str):
            # FIXME support timezone aware strings, e.g. '2023-03-09T17:08:12.4465024+00:00'
            date = datetime.datetime.fromisoformat(date)  # Validate by parsing
        elif type(date) is datetime.date:
            date = datetime.datetime.fromordinal(date.toordinal())
        return datetime.datetime.isoformat(date)

    def register_session(self, ses_path, users=None, file_list=True, **kwargs):
        """
        Register session in Alyx.

        NB: If providing a lab or start_time kwarg, they must match the lab (if there is one)
        and date of the session path.

        Parameters
        ----------
        ses_path : str, pathlib.Path
            The local session path
        users : str, list
            The user(s) to attribute to the session
        file_list : bool, list
            An optional list of file paths to register.  If True, all valid files within the
            session folder are registered.  If False, no files are registered
        location : str
            The optional location within the lab where the experiment takes place
        procedures : str, list
            An optional list of procedures, e.g. 'Behavior training/tasks'
        n_correct_trials : int
            The number of correct trials (optional)
        n_trials : int
            The total number of completed trials (optional)
        json : dict, str
            Optional JSON data
        projects: str, list
            The project(s) to which the experiment belongs (optional)
        type : str
            The experiment type, e.g. 'Experiment', 'Base'
        task_protocol : str
            The task protocol (optional)
        lab : str
            The name of the lab where the session took place.  If None the lab name will be
            taken from the path.  If no lab name is found in the path (i.e. no <lab>/Subjects)
            the default lab on Alyx will be used.
        start_time : str, datetime.datetime
            The precise start time of the session.  The date must match the date in the session
            path.
        end_time : str, datetime.datetime
            The precise end time of the session.

        Returns
        -------
        dict
            An Alyx session record
        list, None
            Alyx file records (or None if file_list is False)

        Raises
        ------
        AssertionError
            Subject does not exist on Alyx or provided start_time does not match date in
            session path.
        ValueError
            The provided lab name does not match the one found in the session path or
            start_time/end_time is not a valid ISO date time.
        requests.HTTPError
            A 400 status code means the submitted data was incorrect (e.g. task_protocol was an
            int instead of a str); A 500 status code means there was a server error.
        ConnectionError
            Failed to connect to Alyx, most likely due to a bad internet connection.
        """
        if isinstance(ses_path, str):
            ses_path = Path(ses_path)
        details = session_path_parts(ses_path.as_posix(), as_dict=True, assert_valid=True)
        # query alyx endpoints for subject, error if not found
        self.assert_exists(details['subject'], 'subjects')

        # look for a session from the same subject, same number on the same day
        with no_cache(self.one.alyx):
            session_id, session = self.one.search(subject=details['subject'],
                                                  date_range=details['date'],
                                                  number=details['number'],
                                                  details=True, query_type='remote')
        users = ensure_list(users or self.one.alyx.user)
        self.assert_exists(users, 'users')

        # if nothing found create a new session in Alyx
        ses_ = {'subject': details['subject'],
                'users': users,
                'type': 'Experiment',
                'number': details['number']}
        if kwargs.get('end_time', False):
            ses_['end_time'] = self.ensure_ISO8601(kwargs.pop('end_time'))
        start_time = self.ensure_ISO8601(kwargs.pop('start_time', details['date']))
        assert start_time[:10] == details['date'], 'start_time doesn\'t match session path'
        if kwargs.get('procedures', False):
            ses_['procedures'] = ensure_list(kwargs.pop('procedures'))
        if kwargs.get('projects', False):
            ses_['projects'] = ensure_list(kwargs.pop('projects'))
        assert ('subject', 'number') not in kwargs
        if 'lab' not in kwargs and details['lab']:
            kwargs.update({'lab': details['lab']})
        elif details['lab'] and kwargs.get('lab', details['lab']) != details['lab']:
            names = (kwargs['lab'], details['lab'])
            raise ValueError('lab kwarg "%s" does not match lab name in path ("%s")' % names)
        ses_.update(kwargs)

        if not session:  # Create from scratch
            ses_['start_time'] = start_time
            session = self.one.alyx.rest('sessions', 'create', data=ses_)
        else:  # Update existing
            if start_time:
                ses_['start_time'] = self.ensure_ISO8601(start_time)
            session = self.one.alyx.rest('sessions', 'update', id=session_id[0], data=ses_)

        _logger.info(session['url'] + ' ')
        # at this point the session has been created. If create only, exit
        if not file_list:
            return session, None
        recs = self.register_files(self.find_files(ses_path) if file_list is True else file_list)
        if recs:  # Update local session data after registering files
            session['data_dataset_session_related'] = ensure_list(recs)
        return session, recs

    def register_files(self, file_list,
                       versions=None, default=True, created_by=None, server_only=False,
                       repository=None, exists=True, dry=False, max_md5_size=None, **kwargs):
        """
        Registers a set of files belonging to a session only on the server.

        Parameters
        ----------
        file_list : list, str, pathlib.Path
            A filepath (or list thereof) of ALF datasets to register to Alyx.
        versions : str, list of str
            Optional version tags.
        default : bool
            Whether to set as default revision (defaults to True).
        created_by : str
            Name of Alyx user (defaults to whoever is logged in to ONE instance).
        server_only : bool
            Will only create file records in the 'online' repositories and skips local repositories
        repository : str
            Name of the repository in Alyx to register to.
        exists : bool
            Whether the files exist on the repository (defaults to True).
        dry : bool
            When true returns POST data for registration endpoint without submitting the data.
        max_md5_size : int
            Maximum file in bytes to compute md5 sum (always compute if None).
        exists : bool
            Whether files exist in the repository. May be set to False when registering files
            before copying to the repository.
        **kwargs
            Extra arguments directly passed as REST request data to /register-files endpoint.

        Returns
        -------
        list of dicts, dict
            A list of newly created Alyx dataset records or the registration data if dry.

        Notes
        -----
        - The registered files may be automatically moved to new revision folders if they are
          protected on Alyx, therefore it's important to check the relative paths of the output.
        - Protected datasets are not checked in dry mode.
        - In most circumstances a new revision will be added automatically, however if this fails
          a 403 HTTP status may be returned.

        Raises
        ------
        requests.exceptions.HTTPError
            Submitted data not valid (400 status code)
            Server side database error (500 status code)
            Revision protected (403 status code)
        """
        F = defaultdict(list)  # empty map whose keys will be session paths
        V = defaultdict(list)  # empty map for versions
        if isinstance(file_list, (str, pathlib.Path)):
            file_list = [file_list]

        if versions is None or isinstance(versions, str):
            versions = itertools.repeat(versions)
        else:
            versions = itertools.cycle(versions)

        # Filter valid files and sort by session
        for fn, ver in zip(map(pathlib.Path, file_list), versions):
            session_path = get_session_path(fn)
            if fn.suffix not in self.file_extensions:
                _logger.debug(f'{fn}: No matching extension "{fn.suffix}" in database')
                continue
            try:
                get_dataset_type(fn, self.dtypes)
            except ValueError as ex:
                _logger.debug('%s', ex.args[0])
                continue
            F[session_path].append(fn.relative_to(session_path))
            V[session_path].append(ver)

        # For each unique session, make a separate POST request
        records = []
        for session_path, files in F.items():
            # this is the generic relative path: subject/yyyy-mm-dd/NNN
            details = session_path_parts(session_path.as_posix(), as_dict=True, assert_valid=True)
            rel_path = PurePosixPath(details['subject'], details['date'], details['number'])
            file_sizes = [session_path.joinpath(fn).stat().st_size for fn in files]
            # computing the md5 can be very long, so this is an option to skip if the file is
            # bigger than a certain threshold
            md5s = [hashfile.md5(session_path.joinpath(fn))
                    if (max_md5_size is None or sz < max_md5_size) else None
                    for fn, sz in zip(files, file_sizes)]

            _logger.info('Registering ' + str(files))

            r_ = {'created_by': created_by or self.one.alyx.user,
                  'path': rel_path.as_posix(),
                  'filenames': [x.as_posix() for x in files],
                  'hashes': md5s,
                  'filesizes': file_sizes,
                  'name': repository,
                  'exists': exists,
                  'server_only': server_only,
                  'default': default,
                  'versions': V[session_path],
                  'check_protected': True,
                  **kwargs
                  }

            # Add optional field
            if details['lab'] and 'labs' not in kwargs:
                r_['labs'] = details['lab']
            # If dry, store POST data, otherwise store resulting file records
            try:
                records.append(r_ if dry else self.one.alyx.post('/register-file', data=r_))
            except requests.exceptions.HTTPError as err:
                # 403 response when datasets already registered and protected by tags
                err_message = err.response.json()
                if not (err_message.get('status_code') == 403 and
                   err_message.get('error') == 'One or more datasets is protected'):
                    raise err  # Some other error occurred; re-raise
                response = err_message['details']
                today_revision = datetime.datetime.today().strftime('%Y-%m-%d')
                new_file_list = []

                for fl, res in zip(files, response):
                    (name, prot_info), = res.items()
                    # Dataset has not yet been registered
                    if not prot_info:
                        new_file_list.append(fl)
                        continue

                    # Check to see if the file path already has a revision in it
                    file_revision = folder_parts(rel_path / fl, as_dict=True)['revision']
                    # Find existing protected revisions
                    existing_revisions = [k for pr in prot_info for k, v in pr.items() if v]

                    if file_revision:
                        # If the revision explicitly defined by the user doesn't exist or
                        # is not protected, register as is
                        if file_revision not in existing_revisions:
                            revision_path = fl.parent
                        else:
                            # Find the next sub-revision that isn't protected
                            new_revision = self._next_revision(file_revision, existing_revisions)
                            revision_path = fl.parent.parent.joinpath(f'#{new_revision}#')

                        if revision_path != fl.parent:
                            session_path.joinpath(revision_path).mkdir(exist_ok=True)
                            _logger.info('Moving %s -> %s', fl, revision_path.joinpath(fl.name))
                            shutil.move(session_path / fl, session_path / revision_path / fl.name)
                        new_file_list.append(revision_path.joinpath(fl.name))
                        continue

                    # The file wasn't in a revision folder but is protected
                    fl_path = fl.parent
                    assert name == fl_path.joinpath(fl.name).as_posix()

                    # Find info about the latest revision
                    # N.B on django side prot_info is sorted by latest revisions first
                    (latest_revision, protected), = prot_info[0].items()

                    # If the latest revision is the original and it is unprotected
                    # no need for revision e.g {'clusters.amp.npy': [{'': False}]}
                    if latest_revision == '' and not protected:
                        # Use original path
                        revision_path = fl_path

                    # If there already is a revision but it is unprotected,
                    # move into this revision folder e.g
                    # {'clusters.amp.npy':
                    #   [{'2022-10-31': False}, {'2022-05-31': True}, {'': True}]}
                    elif not protected:
                        # Check that the latest_revision has the date naming convention we expect
                        # i.e. 'YYYY-MM-DD'
                        try:
                            _ = datetime.datetime.strptime(latest_revision[:10], '%Y-%m-%d')
                            revision_path = fl_path.joinpath(f'#{latest_revision}#')
                        # If it doesn't it probably has been made manually so we don't want to
                        # overwrite this and instead use today's date
                        except ValueError:
                            # NB: It's possible that today's date revision is also protected but is
                            # not the most recent revision. In this case it's safer to let fail.
                            revision_path = fl_path.joinpath(f'#{today_revision}#')

                    # If protected and the latest protected revision is from today we need to make
                    # a sub-revision
                    elif protected and today_revision in latest_revision:
                        if latest_revision == today_revision:  # iterate from appending 'a'
                            new_revision = self._next_revision(today_revision, existing_revisions)
                        else:  # assume the revision is date + character, e.g. '2020-01-01c'
                            alpha = latest_revision[-1]  # iterate from this character
                            new_revision = self._next_revision(
                                today_revision, existing_revisions, alpha)
                        revision_path = fl_path.joinpath(f'#{new_revision}#')

                    # Otherwise cases move into revision from today
                    # e.g {'clusters.amp.npy': [{'': True}]}
                    # e.g {'clusters.amp.npy': [{'2022-10-31': True}, {'': True}]}
                    else:
                        revision_path = fl_path.joinpath(f'#{today_revision}#')

                    # Only move for the cases where a revision folder has been made
                    if revision_path != fl_path:
                        session_path.joinpath(revision_path).mkdir(exist_ok=True)
                        _logger.info('Moving %s -> %s', fl, revision_path.joinpath(fl.name))
                        shutil.move(session_path / fl, session_path / revision_path / fl.name)
                    new_file_list.append(revision_path.joinpath(fl.name))

                assert len(new_file_list) == len(files)
                r_['filenames'] = [p.as_posix() for p in new_file_list]
                r_['filesizes'] = [session_path.joinpath(p).stat().st_size for p in new_file_list]
                r_['check_protected'] = False  # Speed things up by ignoring server-side checks

                records.append(self.one.alyx.post('/register-file', data=r_))
                files = new_file_list

            # Log file names
            _logger.info(f'ALYX REGISTERED DATA {"!DRY!" if dry else ""}: {rel_path}')
            for p in files:
                _logger.info(f'ALYX REGISTERED DATA: {p}')

        return records[0] if len(F.keys()) == 1 else records

    @staticmethod
    def _next_revision(revision: str, reserved: list = None, alpha: str = 'a') -> str:
        """
        Return the next logical revision that is not already in the provided list.
        Revisions will increment by appending a letter to a date or other identifier.

        Parameters
        ----------
        revision : str
            The revision on which to base the new revision.
        reserved : list of str
            A list of reserved (i.e. already existing) revision strings.
        alpha : str
            The starting character as an integer, defaults to 'a'.

        Returns
        -------
        str
            The next logical revision string that's not in the reserved list.

        Examples
        --------
        >>> RegistrationClient._next_revision('2020-01-01')
        '2020-01-01a'
        >>> RegistrationClient._next_revision('2020-01-01', ['2020-01-01a', '2020-01-01b'])
        '2020-01-01c'
        >>> RegistrationClient._next_revision('2020-01-01', ['2020-01-01a', '2020-01-01b'])
        '2020-01-01c'
        """
        if len(alpha) != 1:
            raise TypeError(
                f'`alpha` must be a character; received a string of length {len(alpha)}'
            )
        i = ord(alpha)
        new_revision = revision + chr(i)
        while new_revision in (reserved or []):
            i += 1
            new_revision = revision + chr(i)
        return new_revision

    def register_water_administration(self, subject, volume, **kwargs):
        """
        Register a water administration to Alyx for a given subject

        Parameters
        ----------
        subject : str
            A subject nickname that exists on Alyx
        volume : float
            The total volume administrated in ml
        date_time : str, datetime.datetime, datetime.date
            The time of administration.  If None, the current time is used.
        water_type : str
            A water type that exists in Alyx; default is 'Water'
        user : str
            The user who administrated the water.  Currently logged-in user is the default.
        session : str, UUID, pathlib.Path, dict
            An optional experiment ID to associate
        adlib : bool
            If true, indicates that the subject was given water ad libitum

        Returns
        -------
        dict
            A water administration record

        Raises
        ------
        one.alf.exceptions.AlyxSubjectNotFound
            Subject does not exist on Alyx
        one.alf.exceptions.ALFError
            User does not exist on Alyx
        ValueError
            date_time is not a valid ISO date time or session ID is not valid
        requests.exceptions.HTTPError
            Failed to connect to database, or submitted data not valid (500)
        """
        # Ensure subject exists
        self.assert_exists(subject, 'subjects')
        # Ensure user(s) exist
        user = kwargs.pop('user', self.one.alyx.user)
        self.assert_exists(user, 'users')
        # Ensure volume not zero
        if volume == 0:
            raise ValueError('Water volume must be greater than zero')
        # Post water admin
        wa_ = {
            'subject': subject,
            'date_time': self.ensure_ISO8601(kwargs.pop('date_time', None)),
            'water_administered': float(f'{volume:.4g}'),  # Round to 4 s.f.
            'water_type': kwargs.pop('water_type', 'Water'),
            'user': user,
            'adlib': kwargs.pop('adlib', False)
        }
        # Ensure session is valid; convert to eid
        if kwargs.get('session', False):
            wa_['session'] = self.one.to_eid(kwargs.pop('session'))
            if not wa_['session']:
                raise ValueError('Failed to parse session ID')

        return self.one.alyx.rest('water-administrations', 'create', data=wa_)

    def register_weight(self, subject, weight, date_time=None, user=None):
        """
        Register a subject weight to Alyx.

        Parameters
        ----------
        subject : str
            A subject nickname that exists on Alyx.
        weight : float
            The subject weight in grams.
        date_time : str, datetime.datetime, datetime.date
            The time of weighing.  If None, the current time is used.
        user : str
            The user who performed the weighing.  Currently logged-in user is the default.

        Returns
        -------
        dict
            An Alyx weight record

        Raises
        ------
        one.alf.exceptions.AlyxSubjectNotFound
            Subject does not exist on Alyx
        one.alf.exceptions.ALFError
            User does not exist on Alyx
        ValueError
            date_time is not a valid ISO date time or weight < 1e-4
        requests.exceptions.HTTPError
            Failed to connect to database, or submitted data not valid (500)
        """
        # Ensure subject exists
        self.assert_exists(subject, 'subjects')
        # Ensure user(s) exist
        user = user or self.one.alyx.user
        self.assert_exists(user, 'users')
        # Ensure weight not zero
        if weight == 0:
            raise ValueError('Water volume must be greater than 0')

        # Post water admin
        wei_ = {'subject': subject,
                'date_time': self.ensure_ISO8601(date_time),
                'weight': float(f'{weight:.4g}'),  # Round to 4 s.f.
                'user': user}
        return self.one.alyx.rest('weighings', 'create', data=wei_)
