from setuptools import setup, find_packages


setup(
    name='scluster',
    version='0.0.2',
    author='Yusuke Matsubara',
    author_email='whym@whym.org',
    description='an implementation of spectral clustering for text document collections',
    license='MIT',
    url='https://github.com/whym/scluster',
    packages=find_packages(),
    install_requires=['six'],
    test_suite='tests'
)
