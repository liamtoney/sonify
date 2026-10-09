import subprocess


def test_cli_help():
    subprocess.run(['sonify', '--help'], check=True, stdout=subprocess.DEVNULL)
