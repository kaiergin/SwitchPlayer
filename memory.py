import psutil

def print_memory_diagnostics():
    process = psutil.Process()
    mem_info = process.memory_info()
    # mem_info.rss: Resident Set Size (physical memory)
    # mem_info.vms: Virtual Memory Size
    vm = psutil.virtual_memory()
    print(f"Process Memory: RSS = {mem_info.rss / (1024 * 1024):.2f} MB, VMS = {mem_info.vms / (1024 * 1024):.2f} MB")
    print(f"System Memory: Total = {vm.total / (1024 * 1024):.2f} MB, Available = {vm.available / (1024 * 1024):.2f} MB, Usage = {vm.percent}%")