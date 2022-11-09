"""ALyx File related errors

A set of Alyx and ALF related error classes which provide a more verbose description of the raised
issues.
"""


class ALFError(Exception):
    """A base class for ALF-related errors

    Attributes
    ----------
    explanation : str
        An optional, verbose but general explanation of the error class.  All errors will display
        the same explanation.
    """
    explanation = ''

    def __init__(self, *args, terse=False):
        """A base ALF exception

        Parameters
        ----------
        args : str, any
            A specific error message to display or items to include in message
        terse : bool
            If True, the explanation string is not included in exception message

        Examples
        --------
        >>> raise ALFError("ALF directory doesn't exist")
        one.alf.exceptions.ALFError: ALF directory doesn't exist

        >>> raise ALFError('invalid/path/one', 'invalid/path/two')
        one.alf.exceptions.ALFError: "invalid/path/one", "invalid/path/two"
        """
        if args:
            if len(args) == 1 and isinstance(args[0], str):
                self.message = args[0]
            else:
                self.message = '"' + '", "'.join(map(str, args)) + '"'
        else:
            self.message = ''
        self.terse = terse

    def __str__(self):
        if not self.message and not self.explanation:
            return ''
        return self.message if self.terse else f'{self.message} \n {self.explanation} '


class AlyxSubjectNotFound(ALFError):
    """'Subject not found' error"""
    explanation = 'The subject was not found in Alyx database'


class ALFObjectNotFound(ALFError):
    """'Object not found' error"""
    explanation = ('The ALF object was not found.  This may occur if the object or namespace or '
                   'incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be '
                   'found with the filters `object="trials", namespace="ibl"`')


class ALFMultipleObjectsFound(ALFError):
    """'Multiple objects found' error"""
    explanation = ('Dataset files belonging to more than one object found.  '
                   'ALF names have the pattern '
                   '(_namespace_)object.attribute(_timescale).extension, e.g. for the file '
                   '"_ibl_trials.intervals.npy" the object is "trials"')


class ALFMultipleCollectionsFound(ALFError):
    """'Multiple collections found' error"""
    explanation = ('The matching object/file(s) belong to more than one collection.  '
                   'ALF names have the pattern '
                   'collection/(_namespace_)object.attribute(_timescale).extension, e.g. for the '
                   'file "alf/probe01/spikes.times.npy" the collection is "alf/probe01"')


class ALFMultipleRevisionsFound(ALFError):
    """'Multiple objects found' error"""
    explanation = ('The matching object/file(s) belong to more than one revision.  '
                   'Multiple datasets in different revision folders were found with no default '
                   'specified.')
