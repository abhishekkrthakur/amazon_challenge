

import numpy as np
# needed for include_dirs=[np.get_include()])
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


#===============================================================================
# call in unix terminal in current directory: python setup.py build_ext --inplace
#===============================================================================

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[
                     Extension("cyutils", ["cyutils.pyx"]),
#                     Extension("backwardPyx", ["backwardPyx.pyx"])
                     ],
       include_dirs=[np.get_include()])
	  