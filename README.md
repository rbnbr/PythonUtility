# PythonUtility

Some utility stuff I use from time to time that are project independent.

## os_util

### run_command_on_files_in_dir.py
Runs command provided command (--cmd COMMAND) on all files f_i in provided source dir (--src_dir SOURCE_DIR).
The provided command should be a format string.
In this format string, you can access individual files with {f} (without filetype) and its file type with {t}.
A regex can be specified to only select files in the source dir that match this regex.

*Example:*
- ``python run_command_on_files_in_dir.py --cmd "pyside6-uic.exe ./ui/{f}.{t} -o ./src/ui_{f}.py" --src .\ui\ --regex ".*\.ui$"``

    Compiles all files in ``./ui/`` that end on ``.ui`` using qt's ``pyside6-uic`` compiler and put the result into the directory ``./src/`` with a ``ui_`` prefix and a ``.py`` suffix.
