# Ensures the OpenAlyx password is stored in the parameters so that the user is not prompted for
# the password when instantiating ONE
import one.params as p
from one.tests import TEST_DB_2
par = p.get(silent=True).set('ALYX_PWD', TEST_DB_2['password'])
p.save(par, par.ALYX_URL)
