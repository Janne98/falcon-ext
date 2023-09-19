from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Falcon extension package'
LONG_DESCRIPTION = 'Extension of the falcon-ms package'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="falcon-ext", 
        version=VERSION,
        author="Janne Heirman",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
)