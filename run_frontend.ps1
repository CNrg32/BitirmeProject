$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$projectRoot\mobile"
$command = "`"$projectRoot\.flutter_sdk\bin\flutter.bat`" run -d web-server --web-hostname 127.0.0.1 --web-port 8080 1>> `"$projectRoot\frontend.log`" 2>> `"$projectRoot\frontend.err.log`""
cmd.exe /c $command
