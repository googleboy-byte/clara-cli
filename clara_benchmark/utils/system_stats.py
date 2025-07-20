import os
import platform
import psutil

def get_system_stats():
    """Get comprehensive system statistics"""
    stats = {}
    
    try:
        # Try to import psutil for detailed system stats
        import psutil
        
        # CPU Usage
        stats['cpu_percent'] = psutil.cpu_percent(interval=1)
        stats['cpu_count'] = psutil.cpu_count()
        stats['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
        
        # Memory Usage
        memory = psutil.virtual_memory()
        stats['ram_total'] = f"{memory.total / (1024**3):.2f} GB"
        stats['ram_used'] = f"{memory.used / (1024**3):.2f} GB"
        stats['ram_available'] = f"{memory.available / (1024**3):.2f} GB"
        stats['ram_percent'] = f"{memory.percent:.1f}%"
        
        # Disk Usage
        disk = psutil.disk_usage('/')
        stats['disk_total'] = f"{disk.total / (1024**3):.2f} GB"
        stats['disk_used'] = f"{disk.used / (1024**3):.2f} GB"
        stats['disk_free'] = f"{disk.free / (1024**3):.2f} GB"
        stats['disk_percent'] = f"{disk.percent:.1f}%"
        
        # Network Usage
        network = psutil.net_io_counters()
        stats['network_bytes_sent'] = f"{network.bytes_sent / (1024**2):.2f} MB"
        stats['network_bytes_recv'] = f"{network.bytes_recv / (1024**2):.2f} MB"
        
        # Load Average (Unix-like systems)
        try:
            load_avg = psutil.getloadavg()
            stats['load_avg_1min'] = f"{load_avg[0]:.2f}"
            stats['load_avg_5min'] = f"{load_avg[1]:.2f}"
            stats['load_avg_15min'] = f"{load_avg[2]:.2f}"
        except AttributeError:
            stats['load_avg'] = "N/A"
            
        # CPU Temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get the first available temperature sensor
                for name, entries in temps.items():
                    if entries:
                        stats['cpu_temp'] = f"{entries[0].current:.1f}°C"
                        break
            else:
                stats['cpu_temp'] = "N/A"
        except AttributeError:
            stats['cpu_temp'] = "N/A"
            
    except ImportError:
        # Fallback to basic system information without psutil
        stats['cpu_count'] = os.cpu_count() or "N/A"
        stats['platform'] = platform.platform()
        stats['python_version'] = platform.python_version()
        stats['psutil_available'] = False
        
        # Try to get basic memory info using /proc/meminfo (Linux)
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if 'MemTotal:' in line:
                        total_kb = int(line.split()[1])
                        stats['ram_total'] = f"{total_kb / (1024**2):.2f} GB"
                        break
        except:
            stats['ram_total'] = "N/A"
    
    return stats
