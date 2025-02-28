import os
import concurrent.futures

configs = [
    "AW.json",
    "DC.json",
    "EC.json",
    "pstov1.json",
    "pstov2.json"
]


commands = []
for config in configs:
    commands.append(f" json_name={config}")
print(commands)


def run_command(command, device):
    # Run the command with the device flag
    full_command = f"python3 main.py {command} cuda_device={device}"
    print(f"Running: {full_command}")
    return os.system(full_command), device


def run_commands_in_parallel(commands, max_workers, devices):
    # Dictionary to store the command and the corresponding device
    command_device_map = {i: devices[i % len(devices)] for i in range(len(commands))}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit the initial batch of commands with corresponding devices
        futures = {
            executor.submit(run_command, commands[i], command_device_map[i]): i for i in range(len(commands))
        }

        for future in concurrent.futures.as_completed(futures):
            # Get the index of the command that was run
            index = futures[future]

            try:
                # Retrieve the result (exit code) and device used
                exit_code, device = future.result()
                command = commands[index]
                print(f"Command: {command} finished with exit code {exit_code} on device {device}")
            except Exception as e:
                print(f"Command: {commands[index]} generated an exception: {e}")

            # Remove the finished command from the list
            del commands[index]

            # If there are still commands left, submit the next one with the same device
            if commands:
                next_index = next((i for i in range(len(commands)) if i not in futures.values()), None)
                if next_index is not None:
                    next_command = commands[next_index]
                    futures[executor.submit(run_command, next_command, device)] = next_index


# Assume you have 2 GPUs: devices 0 and 1
devices = [8, 9]

# Set the number of parallel workers based on the number of GPUs
max_workers = len(devices)

run_commands_in_parallel(commands, max_workers, devices)
