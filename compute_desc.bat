@rem compute descriptors from a list of SMILES
set dpath=data
set num=sample

perl raw2phase.pl "%dpath%\Smiles_%num%.txt" "%dpath%\raw_%num%.txt" > %num%.smi
@rem @set /P USR_INPUT_STR="hit any key"
python -m mordred -p 1 %num%.smi -o desc_%num%_woheader.csv
perl add-header.pl desc_%num%_woheader.csv > desc_%num%.csv

