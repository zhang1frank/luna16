from setuptools import setup, find_packages

setup(
    name = "ransack",
    version ="0.1.0",
    install_requires = [
    ],
    package_data = {
        "ransack.cli": [ "*", "packrat/*", "R.modules/*" ],
    },
    entry_points = {
        "console_scripts": [
            "ransack = ransack.cli.app:main"
        ]
    },
    packages = find_packages()
)
