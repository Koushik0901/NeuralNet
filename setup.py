from setuptools import find_packages, setup
setup(
    name='NeuralNet',
    packages=find_packages(),
    version='0.1.0',
    description='Neural Networks from scratch using NumPy',
    author='Me',
    license='MIT',
    install_requires=['numpy', 'tqdm', 'scipy'],
    tests_require=['mypy'],
    test_suite='tests',
)