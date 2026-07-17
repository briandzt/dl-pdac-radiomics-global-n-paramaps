import setuptools

if __name__ == '__main__':
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        # Use pyproject.toml for most configuration
        long_description=long_description,
        long_description_content_type="text/markdown",
        # Package discovery is handled by pyproject.toml
        packages=setuptools.find_packages(where="src"),
        package_dir={"": "src"},
    )
