"""A module for Alyx file related errors
A set of Alyx and ALF related error classes which provide a more verbose description of the raised
issues.
"""


class ALFError(Exception):
    explanation = ''

    def __init__(self, *args, terse=False):
        if args:
            self.message = args[0]
        else:
            self.message = None
        self.terse = terse

    def __str__(self):
        return self.message if self.terse else f"{self.message} \n {self.explanation} "


class AlyxSubjectNotFound(ALFError):
    explanation = 'The subject was not found in Alyx database'


class ALFMultipleObjectsFound(ALFError):
    explanation = ('The search object was not found.  ALF names have the pattern '
                   '(_namespace_)object.attribute(_timescale).extension, e.g. for the file '
                   '"_ibl_trials.intervals.npy" the object is "trials"')


class ALFMultipleCollectionsFound(ALFError):
    explanation = ('The matching object/file(s) belong to more than one collection.  '
                   'ALF names have the pattern '
                   'collection/(_namespace_)object.attribute(_timescale).extension, e.g. for the '
                   'file "alf/probe01/spikes.times.npy" the collection is "alf/probe01"')


class ALFObjectNotFound(ALFError):
    explanation = ('The ALF object was not found.  This may occur if the object or namespace or '
                   'incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be '
                   'found with the filters `object="trials", namespace="ibl"`')
