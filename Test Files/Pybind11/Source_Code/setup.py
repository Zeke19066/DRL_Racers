from distutils.core import setup, Extension

module1 = Extension('keymapper',
                    sources = ['keymapper.cpp'])

setup (name = 'keymapper',
       version = '1.0',
       description = 'Shared Library for Lego Racers keyboard control',
       ext_modules = [module1])

'''

from distutils.core import setup, Extension

module1 = Extension('demo',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = ['/usr/local/include'],
                    libraries = ['tcl83'],
                    library_dirs = ['/usr/local/lib'],
                    sources = ['demo.c'])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       author = 'Martin v. Loewis',
       author_email = 'martin@v.loewis.de',
       url = 'https://docs.python.org/extending/building',
       long_description = 'This is really just a demo package.',
       ext_modules = [module1])

'''