Set-Location -LiteralPath $PSScriptRoot
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_setup.py

