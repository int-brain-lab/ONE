"""
Experiment Ids
==============
There are multiple ways to uniquely identify an experiment:
    - eID (str) : An experiment UUID as a string
    - np (int64) : An experiment UUID encoded as 2 int64s
    - path (Path) : A pathlib ALF path of the form <lab>/Subjects/<subject>/<date>/<number>
    - ref (str) : An experiment reference string of the form yyyy-mm-dd_n_subject
    - url (str) : An remote http session path of the form <lab>/Subjects/<subject>/<date>/<number>

Internally Alyx and ONE uses eID strings to identify sessions.  For example One.search returns a
list of eID strings.  In the ONE cache tables they are represented as a numpy array of 2 int64s
because these are faster to search over.  Session paths, URLs and ref strings are more readable.
"""
from uuid import UUID

from one.api import ONE
import one.alf.io as alfio

one = ONE(base_url='https://openalyx.internationalbrainlab.org')

# One.search returns experiment uuid strings
eids = one.search(data='channels.brainLocation')
assert alfio.is_uuid_string(eids[0])

# eID strings can be easily converted to other forms
session_path = one.eid2path(eids[0])  # returns a pathlib.Path object
assert alfio.is_session_path(session_path)
print(f'Session {"exists" if session_path.exists() else "does not exist"} on disk')

uuid = UUID(eids[0])
assert alfio.is_uuid(uuid)

# These conversion functions can except lists of experiment ids
ref_dict = one.eid2ref(eids)
assert len(ref_dict) == len(eids)
print(ref_dict[0])

# ref strings can be sorted lexicographically (by date, number and subject in that order)
refs = sorted(one.dict2ref(ref_dict))
print(refs)

# Most ids can be interconverted also
eid = one.path2eid(
    one.ref2path(
        one.dict2ref(
            one.eid2ref(eids[0])
        )
    )
)
assert eid == eids[0]

# One load functions can accept most kinds of experiment identifiers
filepath = one.load_dataset(eid, 'channels.brainLocationIds_ccf_2017.npy',
                            download_only=True)
dset = one.load_dataset(session_path, 'channels.brainLocationIds_ccf_2017.npy')
dset = one.load_dataset(filepath, 'channels.brainLocationIds_ccf_2017.npy')
short_path = '/'.join(session_path.parts[-3:])  # 'subject/date/number'
dset = one.load_dataset(short_path, 'channels.brainLocationIds_ccf_2017.npy')
url = one.path2url(filepath)
dset = one.load_dataset(url, 'channels.brainLocationIds_ccf_2017.npy')
dset = one.load_dataset(ref_dict[0], 'channels.brainLocationIds_ccf_2017.npy')
dset = one.load_dataset(refs[0], 'channels.brainLocationIds_ccf_2017.npy')

# Likewise with other load methods...
obj = one.load_object(short_path, 'channels', attribute='brainLocationIds')
