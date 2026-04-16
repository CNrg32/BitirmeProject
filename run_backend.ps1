$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
$command = "set PYTHONPATH=$projectRoot\.vendor;$projectRoot\src&& py -m uvicorn src.main:app --host 127.0.0.1 --port 8000 1>> `"$projectRoot\backend.log`" 2>> `"$projectRoot\backend.err.log`""
cmd.exe /c $command
