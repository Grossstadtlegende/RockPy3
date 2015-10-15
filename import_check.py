__author__ = 'mike'
import RockPy3

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
            RockPy3.tabulate_available = False

    except ImportError:

        try:
            import matplotlib
            RockPy3.logger.info('using matplotlib version %s' % matplotlib.__version__)
        except ImportError:
            RockPy3.logger.error('please install matplotlib version')

        # try:
        #     import lmfit
        #     RockPy3.log.info('using lmfit version %s' % lmfit.__version__)
        # except ImportError:
        #     RockPy3.log.error('please install lmfit version')

        try:
            import pint
            RockPy3.logger.info('using pint version %s' % pint.__version__)
        except ImportError:
            RockPy3.logger.error('please install pint version')
        try:
            import numpy
            RockPy3.logger.info('using numpy version %s' % numpy.__version__)
        except ImportError:
            RockPy3.logger.error('please install numpy version')
        try:
            import scipy
            RockPy3.logger.info('using scipy version %s' % scipy.__version__)
        except ImportError:
            RockPy3.logger.error('please install scipy version')
        try:
            import decorator
            RockPy3.logger.info('using decorator version %s' % decorator.__version__)
        except ImportError:
            RockPy3.logger.error('please install decorator version')

        try:
            import tabulate
            RockPy3.logger.info('using tabulate version {}'.format(tabulate.__version__))
            RockPy3.tabulate_available = True
        except ImportError:
            RockPy3.tabulate_available = False
            RockPy3.logger.warning('Please install module tabulate for nicer output formatting.')
