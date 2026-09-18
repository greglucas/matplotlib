param(
    [Parameter(Mandatory = $true)]
    [string] $OutputPath,
    [int] $IntervalSeconds = 5
)

# Keep this collector deliberately dependency-free so it can run beside pytest
# on a hosted Windows runner.  The output is intended for diagnosing resource
# starvation and subprocess timeouts, not for precise performance accounting.
"timestamp_utc,cpu_percent,memory_used_mb,memory_available_mb,python_processes,total_processes" |
    Set-Content -Path $OutputPath -Encoding utf8

while ($true) {
    $timestamp = [DateTime]::UtcNow.ToString("o")
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        $memoryUsed = [math]::Round(
            ($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1024, 1)
        $memoryAvailable = [math]::Round($os.FreePhysicalMemory / 1024, 1)
    }
    catch {
        $memoryUsed = ""
        $memoryAvailable = ""
    }

    try {
        $cpuSample = Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction Stop
        $cpu = [math]::Round($cpuSample.CounterSamples.CookedValue, 1)
    }
    catch {
        $cpu = ""
    }

    $processes = @(Get-Process -ErrorAction SilentlyContinue)
    $pythonProcesses = @($processes | Where-Object {
        $_.ProcessName -like "python*"
    }).Count
    "$timestamp,$cpu,$memoryUsed,$memoryAvailable,$pythonProcesses,$($processes.Count)" |
        Add-Content -Path $OutputPath -Encoding utf8

    Start-Sleep -Seconds $IntervalSeconds
}
