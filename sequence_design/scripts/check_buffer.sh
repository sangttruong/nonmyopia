echo "Checking mutant v1 with fn v1"
MUTANT_VER=v1 python check_buffer.py --mutant_ver v1 --fn_ver v1

echo "Checking mutant v1 with fn v2"
MUTANT_VER=v1 python check_buffer.py --mutant_ver v1 --fn_ver v2

echo "Checking mutant v2 with fn v1"
MUTANT_VER=v2 python check_buffer.py --mutant_ver v2 --fn_ver v1

echo "Checking mutant v2 with fn v2"
MUTANT_VER=v2 python check_buffer.py --mutant_ver v2 --fn_ver v2

echo "Checking mutant v3 with fn v1"
MUTANT_VER=v3 python check_buffer.py --mutant_ver v3 --fn_ver v1

echo "Checking mutant v3 with fn v2"
MUTANT_VER=v3 python check_buffer.py --mutant_ver v3 --fn_ver v2