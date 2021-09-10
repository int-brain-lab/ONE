# Setting up ONE credentials

```{important}
To set up credentials you will need access to your Alyx username and password in addition to the IBL FlatIron password. 
If you do not have access to these, please get in contact with a member of the IBL software team.
```

In order to use the ONE interface to access IBL internal data, it is necessary to provide some credentials that allow ONE to 
connect to the Alyx database and the FlatIron server. These credentials are stored locally on your computer in a 
parameter file. 

In a python terminal, type:

```python
from one.api import ONE
one = ONE(base_url='https://alyx.internationalbrainlab.org')
```

You will then be prompted to enter information in the following order. 
 
  
```
ALYX_LOGIN:             # Input your Alyx username
HTTP_DATA_SERVER:       # Keep default - should be automatically set as: https://ibl.flatironinstitute.org
HTTP_DATA_SERVER_LOGIN: # Input FlatIron username
Alyx password:          # Input your Alyx password
FlatIron HTTP password:	# Input FlatIron password
Location of download cache: # Keep default
Would you like to set URL as defualt one?: # Enter y
```

The entries that you will need to change from default are: `ALYX_LOGIN`, `Alyx password`, `HTTP_DATA_SERVER_LOGIN` and 
`FlatIron HTTP password`. For the remaining entries keep the default values by pressing
the Enter key.
