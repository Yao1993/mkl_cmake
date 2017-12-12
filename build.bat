call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.0.124\windows\mkl\bin\mklvars.bat" intel64
cd build
del * /s /q
FOR /D %%p IN ("./*.*") DO rmdir "%%p" /s /q
cmake -G "Visual Studio 15 2017 Win64" ..
cd ..