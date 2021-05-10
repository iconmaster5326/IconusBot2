import setuptools

setuptools.setup(
    name="iconusbot",
    version="0.0.0",
    classifiers=["Programming Language :: Python :: 3"],
    package_dir={"": "iconusbot"},
    packages=setuptools.find_packages(where="iconusbot"),
    entry_points={"console_scripts": ["iconusbot=iconusbot.__main__:main"]},
    install_requires=["lark", "discord.py", "pyyaml", "plotly", "kaleido", "pandas"],
)
