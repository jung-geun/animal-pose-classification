@echo off
set root=%USERPROFILE%\anaconda3
set path=%root%\Scripts;%root%\Library\bin;%root%\Library\mingw-w64\bin;%root%\Library\usr\bin;

call %root%\Scripts\activate.bat %root%

call conda create -n yoga python=3.10 --yes
call conda activate yoga
call pip install -r requirements.txt
