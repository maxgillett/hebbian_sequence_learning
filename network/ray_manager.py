import subprocess
from pssh.clients import ParallelSSHClient
from gevent import joinall

import argparse

hosts = {
    #'Subhadra': '10.122.160.27',
    'Alex': '10.122.160.25',
    #'Alessandro': '10.122.160.35',
    'Achint1': '10.122.160.34',  
    'Achint2': '10.122.160.21',

    # Intentionally disabled
    #'Chicago1': '10.122.162.118',
    #'Chicago2': '10.122.160.28',
    #'Chicago3': '10.122.163.34',
    #'Chicago4': '10.122.162.69',

    # Idle-only
    #'Stan': '10.122.160.23',

    # Not working
    #'Chicago5': '10.122.162.166',
}

hosts2 = {
}

parser = argparse.ArgumentParser(description='Manage Ray workers')
parser.add_argument('--up', dest='up', action='store_true', help='Start ray workers')
parser.add_argument('--down', dest='down', action='store_true', help='Stop ray workers')
parser.add_argument('--run', dest='run', action='store_true', help='Run command on workers')
parser.add_argument('--cmd', dest='cmd', help='Command to run')
args = parser.parse_args()

print(list(hosts.values()))
client1 = ParallelSSHClient(list(hosts.values()), user='mhg19')
client2 = ParallelSSHClient(list(hosts2.values()), user='mhg19')

def print_output(output):
    for host, host_output in output.items():
        for line in host_output.stdout:
            print("stdout:", line)
        for line in host_output.stderr:
            print("stderr:", line)

def start_master():
    subprocess.check_output(['ray start --head --redis-port=6382 --include-webui'], shell=True)

def stop_master():
    subprocess.check_output(['ray stop'], shell=True)

def start_workers():
    output = client1.run_command(
            'source ~/venv/bin/activate; ray start --address=10.122.160.26:6382')
    output = client2.run_command(
            'source ~/venv/bin/activate; ray start --address=10.122.160.26:6382')
    print_output(output)

def stop_workers():
    output = client1.run_command('source ~/venv/bin/activate; ray stop')
    print_output(output)
    output = client2.run_command('source ~/venv/bin/activate; ray stop')
    print_output(output)

def run_command(cmd):
    output = client1.run_command('source ~/venv/bin/activate; '+cmd)
    print_output(output)
    output = client2.run_command('source ~/venv/bin/activate; '+cmd)
    print_output(output)

if args.run:
    run_command(args.cmd)

if args.up:
    stop_master()
    start_master()
    stop_workers()
    start_workers()

if args.down:
    stop_workers()
    stop_master()
