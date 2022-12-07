@echo off

call wget.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
call Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%cd%\Miniconda3

set root=%cd%\Miniconda3
set path=%root%\Scripts;%root%\Library\bin;%root%\Library\mingw-w64\bin;%root%\Library\usr\bin;

call %root%\Scripts\activate.bat %root%

call conda create -n yoga python=3.10 --yes
call conda activate yoga
call pip install -r requirements.txt
