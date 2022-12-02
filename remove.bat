@echo off
set root=%USERPROFILE%\anaconda3
set path=%root%\Scripts;%root%\Library\bin;%root%\Library\mingw-w64\bin;%root%\Library\usr\bin;

call conda activate
call conda env remove -n yoga --yes
