import warnings
warnings.filterwarnings('always', category=DeprecationWarning, module='one')
try:
    import iblutil
except ModuleNotFoundError:
    warnings.warn('Module not found: please run `pip install iblutil`')
