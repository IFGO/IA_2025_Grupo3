from setuptools import setup, find_packages

setup(
    name='mlp_crypto',
    version='0.1.0',
    author='Seu Nome',
    author_email='seuemail@example.com',
    description='Projeto de previsão de preços de criptomoedas usando MLP',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',  # ou 'torch', dependendo da biblioteca que você está usando
        'matplotlib',
        'pytest',
        'pytest-cov'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)