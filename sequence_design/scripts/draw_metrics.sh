echo "Drawing mutant v1 with fn v1"
MUTANT_VER=v1 python draw_metrics.py --mutant_ver v1 --fn_ver v1

echo "Drawing mutant v1 with fn v2"
MUTANT_VER=v1 python draw_metrics.py --mutant_ver v1 --fn_ver v2

echo "Drawing mutant v2 with fn v1" 
MUTANT_VER=v2 python draw_metrics.py --mutant_ver v2 --fn_ver v1

echo "Drawing mutant v2 with fn v2"
MUTANT_VER=v2 python draw_metrics.py --mutant_ver v2 --fn_ver v2

echo "Drawing mutant v3 with fn v1"
MUTANT_VER=v3 python draw_metrics.py --mutant_ver v3 --fn_ver v1

echo "Drawing mutant v3 with fn v2"
MUTANT_VER=v3 python draw_metrics.py --mutant_ver v3 --fn_ver v2