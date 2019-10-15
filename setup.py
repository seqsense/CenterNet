from setuptools import setup, find_packages

setup(
        name='Centernet',
        package_dir={'Centernet': 'src'},
        packages=['Centernet'],
        install_requires=[
            'opencv-python',
            'Cython',
            'numba',
            'progress',
            'matplotlib',
            'easydict',
            'scipy',
            'torch',
            'torchvision',
        ],
)
