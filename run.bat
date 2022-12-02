@echo off

set root=%USERPROFILE%\anaconda3
set path=%root%\Scripts;%root%\Library\bin;%root%\Library\mingw-w64\bin;%root%\Library\usr\bin;

call %root%\Scripts\activate.bat %root%

call conda activate yoga
start %root%\envs\yoga\python.exe app.py
