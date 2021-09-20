# FAQ
## How do I release my own data with the ONE API?
First create a directory of [ALF datasets](notebooks/datasets_and_types), generate the
cache tables, then share the directory with others.  See [data sharing](notebooks/data_sharing)
guide for more information.

## How do I use ONE without connecting to my database?
If you temporarily lose access to your Alyx database, you can instantiate ONE in local mode:
```
one = ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local')
```
Read more about ONE modes [here](notebooks/one_modes)
