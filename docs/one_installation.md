# ONE installation and setup

## 1. Installation
ONE can be installed as a standalone package with python 3.7 or later by running,
```
pip install ONE-api
```

```{note}
If you have installed the unified IBL environment, ONE will already have been automatically
installed, so you can skip this step!
```

## 2. Setup
To start using ONE, we must first configure some settings that tell ONE whether it should connect to a database or
use a local file system

### Create an account on the public database
The public IBL database, [Open Alyx](https://openalyx.internationalbrainlab.org), needs an account.
It is free, it takes a minute, and it is yours: accounts survive data releases, so you register once
and keep the same credentials as new data is published.

Go to [openalyx.internationalbrainlab.org/signup](https://openalyx.internationalbrainlab.org/signup)
and either

- **register with an email address**, choosing a username and password. You will be sent a link to
  confirm the address before the account becomes active; or
- **sign in with [ORCiD](https://orcid.org)**, if you have an ORCiD iD. No password is involved -
  ORCiD vouches for you - so see
  [signing in without a password](#signing-in-without-a-password) below for how to authenticate ONE.

```{note}
Your account is private to you. Public accounts cannot see other users of the database.
```

### Connect to IBL Public database
By default ONE is configured to connect to the public IBL database, this can be setup by typing the following
```python
from one.api import ONE
ONE.setup(silent=True)  # silent means use default parameters for the IBL public database
one = ONE()  # you will be asked for your username, then your password
```

The first call asks for the username you registered with and then for your password. Both are
remembered afterwards - ONE stores an access token, not your password - so later sessions are just
`one = ONE()`. To avoid the username prompt entirely, pass it to the setup:

```python
ONE.setup(silent=True, username='cajal')
```

```{warning}
ONE writes these settings to a parameter file for you. Do not edit that file by hand; re-run
`ONE.setup()` instead.
```

(signing-in-without-a-password)=
### Signing in without a password
An account created through ORCiD has no password, so there is nothing to type at the password
prompt. Such an account authenticates with an **API token** instead:

1. Sign in to [openalyx.internationalbrainlab.org](https://openalyx.internationalbrainlab.org) and
   open the [/me](https://openalyx.internationalbrainlab.org/me) page, which shows your token.
2. Run `one = ONE()`, leave the password prompt blank, and paste the token when asked for one.

The token is a credential: treat it like a password, and regenerate it from the same page if it
leaks. In a script, or anywhere there is no one to answer a prompt, pass it directly - the database
is asked whose token it is, so the username is optional:

```python
from one.api import ONE
one = ONE(token='a1b2c3...')
```

If you are having an issue at this stage, or need to re-configure ONE, please see [this FAQ page](/FAQ.md#i-made-a-mistake-during-setup-and-now-can-t-call-setup-how-do-i-fix-it).

### Using local folder structure
ONE can also be used independently of a database by using a local cache directory. This can be setup in the following way

```python
from one.api import One
one = One(cache_dir='/home/user/downlaods/ONE/behavior_paper')
```

For more information about using ONE with a local cache directory please refer to this [section](notebooks/data_sharing).

### Connecting to specific database (relevant for IBL users)
To connect to a specific database, for example the internal IBL Alyx database, a base-url argument must be given
```python
from one.api import ONE
one = ONE(base_url='https://alyx.internationalbrainlab.org')
```
```{note}
When connecting to the IBL internal database you will be prompted to enter credentials. See the IBL Welcome Guide for more information.
```

```{Warning}
The internal IBL database is only open to IBL members, please connect to our public database to access our publically
available datasets
```

## 3. Post setup
Once you've setup the server you can initialise ONE in the following way and it will automatically connect to the
default database configured in your settings,
```python
from one.api import ONE
one = ONE()
```

To change your default database, or re-run the setup for a given database, you can use the following

```python
ONE.setup(base_url='https://test.alyx.internationalbrainlab.org', make_default=True)
```

## 4. Update
To update, open a shell terminal, activate the environment `iblenv` and launch the command:
```python
pip install -U ONE-api
```

You can check the API version with Python one of two ways:
```python
from one.api import ONE
print(ONE.version)

from one import __version__
print(__version__)
```
