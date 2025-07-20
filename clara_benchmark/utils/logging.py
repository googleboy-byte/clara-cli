import time
import os

from clara_benchmark.utils.system_stats import get_system_stats

def log(message, log_file, call_log_main=True, overwrite_last_line=False):
    time_message = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(time_message + " " + message + "\n")
    log_message = "LOG: " + time_message + " " + message
    if call_log_main:
        log_main(log_message, stdout=False)
    if overwrite_last_line:
        with open(log_file, "r") as f:
            lines = f.readlines()
        with open(log_file, "w") as f:
            f.writelines(lines[:-1])
            f.write(log_message + "\n")

def log_main(message, log_file="./logs/main.log", stdout=True, error=False, system_stats=True):
    time_message = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(time_message + " " + message + "\n")
    if stdout:
        if error:
            print("ERROR: " + time_message + " " + message)
        else:
            print("MAIN: " + time_message + " " + message)
    if system_stats:
        log_system_status(message, stdout=stdout, error=error)

def log_system_status(message, log_file="./logs/system_status.log", stdout=True, error=False):
    time_message = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Get system statistics
    stats = get_system_stats()
    
    # Format the system stats message
    stats_message = f"SYSTEM STATS - "
    if 'psutil_available' in stats and not stats['psutil_available']:
        stats_message += f"CPU Cores: {stats['cpu_count']}, Platform: {stats['platform']}, "
        stats_message += f"Python: {stats['python_version']}, RAM Total: {stats['ram_total']}"
    else:
        stats_message += f"CPU: {stats['cpu_percent']}% ({stats['cpu_count']} cores, {stats['cpu_freq']} MHz), "
        stats_message += f"RAM: {stats['ram_percent']} ({stats['ram_used']}/{stats['ram_total']}), "
        stats_message += f"Disk: {stats['disk_percent']} ({stats['disk_used']}/{stats['disk_total']}), "
        stats_message += f"Network: ↑{stats['network_bytes_sent']} ↓{stats['network_bytes_recv']}"
        
        if 'cpu_temp' in stats and stats['cpu_temp'] != "N/A":
            stats_message += f", Temp: {stats['cpu_temp']}"
        
        if 'load_avg_1min' in stats:
            stats_message += f", Load: {stats['load_avg_1min']}/{stats['load_avg_5min']}/{stats['load_avg_15min']}"
    
    # Combine user message with system stats
    full_message = f"{message} \n{stats_message} \n\n"
    
    with open(log_file, "a") as f:
        f.write(time_message + " " + full_message + "\n")
    
    if stdout:
        if error:
            print("\nERROR: " + time_message + " " + stats_message + "\n")
        else:
            print("\nSYSTEM: " + time_message + " " + stats_message + "\n")
    
    