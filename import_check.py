__author__ = 'mike'
import RockPy3
import pip


def check_imports(): #todo is pip always installed?
    try:
        import pip
        package_version = {
                               pkg.key:pkg.version
                               for pkg in pip.get_installed_distributions()
                               if pkg.key in RockPy3.dependencies
                               }

        for package in RockPy3.dependencies:
            if package not in package_version:
                RockPy3.logger.error('please install %s' %package)
            else:
                RockPy3.logger.info('using {: <12}: version {}'.format(package, package_version[package]))
        if 'tabulate' not in package_version:
            RockPy3.logger.warning('Please install module tabulate for nicer output formatting.')

    except ImportError:

        import matplotlib
        RockPy3.logger.info('using matplotlib version %s' % matplotlib.__version__)
        import numpy
        RockPy3.logger.info('using numpy version %s' % numpy.__version__)
        import pint
        RockPy3.logger.info('using pint version %s' % pint.__version__)

        try:
            from tabulate import tabulate
            RockPy3.logger.info('using tabulate version {}'.format(tabulate.__version__))
            RockPy3.tabulate_available = True
        except ImportError:
            RockPy3.tabulate_available = False
            RockPy3.logger.warning('Please install module tabulate for nicer output formatting.')
