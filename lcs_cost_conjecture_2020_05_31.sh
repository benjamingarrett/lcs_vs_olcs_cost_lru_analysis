f1=lcs1_cost_0011n_ceil_n_b/lcs1_cost_m_0011n_ceil_n_3.csv
#f2=lcs2_cost_0011n_ceil_n_b/lcs2_cost_m_0011n_ceil_n_3.csv
f2=../lcs2_traceback_0011n_ceil_n_b/m_0011n_ceil_n_3.csv
f3=olcs4_cost_0011n_ceil_n_b/olcs4_cost_m_0011n_ceil_n_3.csv
python3 lcs_cost_conjecture_2020_05_31.py $f1 $f2 $f3 10 300 2000
