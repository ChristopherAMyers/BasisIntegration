from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import platform


ext_modules = [
        Extension(
            "integration",
            ["integration.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
        )
    ]

print(ext_modules)
extensions=cythonize(ext_modules)

#extensions=cythonize('integration.pyx', compiler_directives={'language_level' : "3"})

setup(ext_modules=extensions)
