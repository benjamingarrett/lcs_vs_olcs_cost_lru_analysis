# Usage: <folder1> <folder2> <b>
# Assumes n,csize,misses
# Conjecture about csize as function of n


import os, sys, matplotlib.pyplot as plt


def get_list(fn):
  print('fn {}'.format(fn))
  return [ln.rstrip('\n').split(',') for ln in open(fn)]


def points_dict(folder, files, col):
  k = 0
  d = {}
  while k < len(files):
    tmp_str = files[k][15:]
    b = int(tmp_str[0:tmp_str.find('.')])
    d[b] = []
    lst = get_list(folder+'/'+files[k])
    j = 1   # skip heading
    while j < len(lst):
      d[b].append((int(lst[j][0]), int(lst[j][col])))
      j += 1
    k += 1
  return d


def points_from_csv(fname, col=1, has_header=False):
  print('fn {}'.format(fname))
  lst = get_list(fname)
  if has_header==True:
    lst = lst[1:]
  return [(int(k[0]), int(k[col])) for k in lst]


print('args: {}'.format(sys.argv))
if len(sys.argv) < 4:
  print('Usage: <csv file> <csv file> <csv file>')
  exit()
pts1 = points_from_csv(sys.argv[1], has_header=True)
pts1 = [(k[0], k[1]) for k in pts1 if k[0] <= 200]
label1 = 'LCS 1 cost'

pts2 = points_from_csv(sys.argv[2], has_header=True)
pts2 = [(k[0], k[1]) for k in pts2 if k[0] <= 200]
label2 = 'LCS 2 cost'

pts3 = points_from_csv(sys.argv[3], has_header=True)
pts3 = [(k[0], k[1]) for k in pts3 if k[0] <= 200]
label3 = 'OLCS 4 cost'

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([k[0] for k in pts1], [k[1] for k in pts1], '*', label=label1)
ax.plot([k[0] for k in pts2], [k[1] for k in pts2], '*', label=label2)
ax.plot([k[0] for k in pts3], [k[1] for k in pts3], '*', label=label3)
plt.xlabel('problem size (n)')
plt.ylabel('critical cache size')
t = 'LCS cost vs OLCS traceback, LRU, critical cache size (csize)'
t+= '\nInstance type: a = ceiling(n/3)'
t+= '\n1st sequence: 0^a 1^a 2^(n-2a)'
t+= '\n2nd sequence: 2^(n-2a) 1^a 0^a'
ax.legend(loc='upper left')
plt.title(t)
plt.show()
