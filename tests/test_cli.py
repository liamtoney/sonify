import os
import subprocess

import pytest


def test_cli_help():
    subprocess.run(['sonify', '--help'], check=True, stdout=subprocess.DEVNULL)


@pytest.mark.skipif(
    not os.environ.get('GITHUB_SHA'), reason='must be run on GitHub Actions'
)
def test_cli_version():
    output = subprocess.run(
        ['sonify', '--version'], capture_output=True, text=True
    ).stdout.strip()
    assert output == 'sonify, rev. 0+g{}'.format(os.environ['GITHUB_SHA'][:7])
