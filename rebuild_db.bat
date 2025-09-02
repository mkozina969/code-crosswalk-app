@echo off
echo Rebuilding crosswalk.db from crosswalk.csv ...
python make_crosswalk_db.py --csv crosswalk.csv --db data\crosswalk.db --rebuild
echo Done! Database is ready at data\crosswalk.db
pause
