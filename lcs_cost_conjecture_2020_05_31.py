# Usage: <folder1> <folder2> <b>
# Assumes n,csize,misses
# Conjecture about csize as function of n


import os, sys, matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score


MAX_ITERATIONS = 5000000


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


def running_max(pts):
  j = 1
  lst = [(pts[0][0], pts[0][1])]
  while j < len(pts):
    if pts[j][1] > lst[j-1][1]:
      lst.append((pts[j][0], pts[j][1]))
    else:
      lst.append((pts[j][0], lst[j-1][1]))
    j += 1
  return lst


def left_endpoints(pts):
  lst = [(pts[0][0], pts[0][1])]
  j = 1
  while j < len(pts):
    if pts[j][1] > pts[j-1][1]:
      lst.append((pts[j][0], pts[j][1]))
    j += 1
  return lst


def right_endpoints(pts):
  lst = [(pts[0][0], pts[0][1])]
  j = 1
  while j < len(pts):
    if pts[j][1] > pts[j-1][1]:
      lst.append((pts[j-1][0], pts[j-1][1]))
    j += 1
  return lst


def get_best_poly_fit(pts, n0, n1):
  cjpts = [(k[0], k[1]) for k in pts if n0 <= k[0] and k[0] <= n1]
  x = [k[0] for k in cjpts]
  y = [k[1] for k in cjpts]
  p = np.poly1d(np.polyfit(x, y, 2))
  r2 = r2_score(y, p(x))
  return p, r2


def get_poly_fit_pts(poly, n0, n1):
  x = np.linspace(n0, n1, 100)
  y = poly(x)
  pts = []
  j = 0
  while j < len(x):
    pts.append((x[j], y[j]))
    j += 1
  return pts


def get_poly_pts(c2, c1, c0, n0, n1):
  x = np.linspace(n0, n1, 100)
  f = lambda x: c2 * x**2 + c1 * x + c0
  y = f(x)
  pts = []
  j = 0
  while j < len(x):
    pts.append((x[j], y[j]))
    j += 1
  return pts


def get_poly_fit_str(poly):
  s = ''
  j = 0
  while j < poly.o:
    exp = ' x'
    if poly.o - j > 1:
      exp += '^{}'.format(poly.o - j)
    if poly.c[j+1] >= 0:
      s += '{} {} +'.format(poly.c[j], exp)
    else:
      s += '{} {} '.format(poly.c[j], exp)
    j += 1
  s += '{}'.format(poly.c[poly.o])
  return s


def get_poly_str(poly_bound_c2, poly_bound_c1, poly_bound_c0):
  s = '{} x^2 '.format(poly_bound_c2)
  if poly_bound_c1 >= 0:
    s += '+{} x '.format(poly_bound_c1)
  else:
    s += '{} x '.format(poly_bound_c1)
  if poly_bound_c0 >= 0:
    s += '+{}'.format(poly_bound_c0)
  else:
    s += '{}'.format(poly_bound_c0)
  return s


def get_best_poly_bound_0(pts, n0, n1, eps, bound_func):
  cjpts = [(k[0], k[1]) for k in pts if n0 <= k[0] and k[0] <= n1]
  x = [k[0] for k in cjpts]
  y = [k[1] for k in cjpts]
  p = np.poly1d(np.polyfit(x, y, 2))
  r2 = r2_score(y, p(x))
  cnt = 0
  c2 = p.c[0]
  c1 = p.c[1]
  c0 = p.c[2]
  bound = lambda x: c2 * x**2 + c1 * x + c0
  while not bound_func(cjpts, bound):
    c0 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  return c2, c1, c0


def get_best_poly_bound_1(pts, n0, n1, eps, bound_func):
  cjpts = [(k[0], k[1]) for k in pts if n0 <= k[0] and k[0] <= n1]
  x = [k[0] for k in cjpts]
  y = [k[1] for k in cjpts]
  p = np.poly1d(np.polyfit(x, y, 2))
  r2 = r2_score(y, p(x))
  cnt = 0
  c2 = p.c[0]
  c1 = p.c[1]
  c0 = p.c[2]
  bound = lambda x: c2 * x**2 + c1 * x + c0
  while not bound_func(cjpts, bound):
    c1 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  return c2, c1, c0


def get_best_poly_bound_2(pts, n0, n1, eps, bound_func):
  cjpts = [(k[0], k[1]) for k in pts if n0 <= k[0] and k[0] <= n1]
  x = [k[0] for k in cjpts]
  y = [k[1] for k in cjpts]
  p = np.poly1d(np.polyfit(x, y, 2))
  r2 = r2_score(y, p(x))
  cnt = 0
  c2 = p.c[0]
  c1 = p.c[1]
  c0 = p.c[2]
  bound = lambda x: c2 * x**2 + c1 * x + c0
  while not bound_func(cjpts, bound):
    c2 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  return c2, c1, c0


def get_best_poly_bound_3(pts, n0, n1, eps, bound_func, dependent_convergence=False):
  cjpts = [(k[0], k[1]) for k in pts if n0 <= k[0] and k[0] <= n1]
  x = [k[0] for k in cjpts]
  y = [k[1] for k in cjpts]
  p = np.poly1d(np.polyfit(x, y, 2))
  r2 = r2_score(y, p(x))
  cnt = 0
  c2 = p.c[0]
  c1 = p.c[1]
  c0 = c0_orig = p.c[2]
  bound = lambda x: c2 * x**2 + c1 * x + c0
  while not bound_func(cjpts, bound):
    c0 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  if dependent_convergence:
    bound = lambda x: c2 * x**2 + c1 * x + c0
  else:
    bound = lambda x: c2 * x**2 + c1 * x + c0_orig
  while not bound_func(cjpts, bound):
    c1 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  return c2, c1, c0


def get_best_poly_bound_4(pts, n0, n1, eps, bound_func, dependent_convergence=False):
  cjpts = [(k[0], k[1]) for k in pts if n0 <= k[0] and k[0] <= n1]
  x = [k[0] for k in cjpts]
  y = [k[1] for k in cjpts]
  p = np.poly1d(np.polyfit(x, y, 2))
  r2 = r2_score(y, p(x))
  cnt = 0
  c2 = p.c[0]
  c1 = p.c[1]
  c0 = c0_orig = p.c[2]
  bound = lambda x: c2 * x**2 + c1 * x + c0
  while not bound_func(cjpts, bound):
    c0 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  if dependent_convergence:
    bound = lambda x: c2 * x**2 + c1 * x + c0
  else:
    bound = lambda x: c2 * x**2 + c1 * x + c0_orig
  while not bound_func(cjpts, bound):
    c2 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  return c2, c1, c0


def get_best_poly_bound_5(pts, n0, n1, eps, bound_func, dependent_convergence=False):
  cjpts = [(k[0], k[1]) for k in pts if n0 <= k[0] and k[0] <= n1]
  x = [k[0] for k in cjpts]
  y = [k[1] for k in cjpts]
  p = np.poly1d(np.polyfit(x, y, 2))
  r2 = r2_score(y, p(x))
  cnt = 0
  c2 = p.c[0]
  c1 = c1_orig = p.c[1]
  c0 = p.c[2]
  bound = lambda x: c2 * x**2 + c1 * x + c0
  while not bound_func(cjpts, bound):
    c1 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  if dependent_convergence:
    bound = lambda x: c2 * x**2 + c1 * x + c0
  else: 
    bound = lambda x: c2 * x**2 + c1_orig * x + c0
  while not bound_func(cjpts, bound):
    c2 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  return c2, c1, c0


def get_best_poly_bound_6(pts, n0, n1, eps, bound_func, dependent_convergence=False):
  cjpts = [(k[0], k[1]) for k in pts if n0 <= k[0] and k[0] <= n1]
  x = [k[0] for k in cjpts]
  y = [k[1] for k in cjpts]
  p = np.poly1d(np.polyfit(x, y, 2))
  r2 = r2_score(y, p(x))
  cnt = 0
  c2 = p.c[0]
  c1 = c1_orig = p.c[1]
  c0 = c0_orig = p.c[2]
  bound = lambda x: c2 * x**2 + c1 * x + c0
  while not bound_func(cjpts, bound):
    c0 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  if dependent_convergence:
    bound = lambda x: c2 * x**2 + c1 * x + c0
  else: 
    bound = lambda x: c2 * x**2 + c1 * x + c0_orig
  while not bound_func(cjpts, bound):
    c1 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  if dependent_convergence:
    bound = lambda x: c2 * x**2 + c1 * x + c0
  else: 
    bound = lambda x: c2 * x**2 + c1_orig * x + c0_orig
  while not bound_func(cjpts, bound):
    c2 += eps
    cnt += 1
    if cnt > MAX_ITERATIONS:
      print('exiting before convergence')
      exit()
  return c2, c1, c0


def conjecture_points(min_x, max_x, f):
  return [(k, f(k)) for k in range(min_x, max_x+1)]


def bounds_above(pts, f):
  j = 0
  while j < len(pts):
    if float(pts[j][1]) >= f(pts[j][0]):
      return False
    j += 1
  return True


def bounds_below(pts, f):
  j = 0
  while j < len(pts):
    if float(pts[j][1]) <= f(pts[j][0]):
      return False
    j += 1
  return True


def points_from_csv(fname, col=1, has_header=False):
  print('fn {}'.format(fname))
  lst = get_list(fname)
  if has_header==True:
    lst = lst[1:]
  return [(int(k[0]), int(k[col])) for k in lst]





print('args: {}'.format(sys.argv))
if len(sys.argv) < 7:
  print('Usage: <csv file> <csv file> <csv file> <min_x> <max_x> <confirm_to_x>')
  exit()
min_x = int(sys.argv[4])
max_x = int(sys.argv[5])
confirm_to_x = int(sys.argv[6])

pts1 = points_from_csv(sys.argv[1], has_header=True)
pts1 = [(k[0], k[1]) for k in pts1]
label1 = 'LCS 1 cost'

pts2 = points_from_csv(sys.argv[2], has_header=True)
pts2 = [(k[0], k[1]) for k in pts2]
label2 = 'LCS 2 cost'

pts3 = points_from_csv(sys.argv[3], has_header=True)
pts3 = [(k[0], k[1]) for k in pts3]
label3 = 'OLCS 4 cost'

if 'comparison plot' and False:
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

# now estimate growth rate of the better version
data_pts = pts2
rmx = running_max(data_pts)
lft = left_endpoints(rmx)
rgh = right_endpoints(rmx)

confirm_pts = [(k[0], k[1]) for k in rmx if min_x <= k[0] and k[0] <= confirm_to_x]
confirm_x = [k[0] for k in confirm_pts]
confirm_y = [k[1] for k in confirm_pts]

poly_lft_obj, r2_lft = get_best_poly_fit(lft, min_x, max_x)
poly_lft_pts = get_poly_fit_pts(poly_lft_obj, min_x, confirm_to_x)
poly_lft_str = get_poly_fit_str(poly_lft_obj)

poly_rgh_obj, r2_lft = get_best_poly_fit(rgh, min_x, max_x)
poly_rgh_pts = get_poly_fit_pts(poly_rgh_obj, min_x, confirm_to_x)
poly_rgh_str = get_poly_fit_str(poly_rgh_obj)


# left endpoint upper bound
eps = 0.0001
poly_upper_bound_c2, poly_upper_bound_c1, poly_upper_bound_c0 = get_best_poly_bound_0(lft, min_x, max_x, eps, bounds_above)
poly_upper_bound_pts = get_poly_pts(poly_upper_bound_c2, poly_upper_bound_c1, poly_upper_bound_c0, min_x, confirm_to_x)
poly_upper_bound_str = get_poly_str(poly_upper_bound_c2, poly_upper_bound_c1, poly_upper_bound_c0)
poly_upper_bound = lambda x: poly_upper_bound_c2 * x**2 + poly_upper_bound_c1 * x + poly_upper_bound_c0
poly_upper_bound_confirmed = bounds_above(confirm_pts, poly_upper_bound)
if poly_upper_bound_confirmed == False:
  poly_upper_bound_c2, poly_upper_bound_c1, poly_upper_bound_c0 = get_best_poly_bound_1(lft, min_x, max_x, eps, bounds_above)
  poly_upper_bound_pts = get_poly_pts(poly_upper_bound_c2, poly_upper_bound_c1, poly_upper_bound_c0, min_x, confirm_to_x)
  poly_upper_bound_str = get_poly_str(poly_upper_bound_c2, poly_upper_bound_c1, poly_upper_bound_c0)
  poly_upper_bound = lambda x: poly_upper_bound_c2 * x**2 + poly_upper_bound_c1 * x + poly_upper_bound_c0
  poly_upper_bound_confirmed = bounds_above(confirm_pts, poly_upper_bound) 
if poly_upper_bound_confirmed == False:
  poly_upper_bound_c2, poly_upper_bound_c1, poly_upper_bound_c0 = get_best_poly_bound_2(lft, min_x, max_x, eps, bounds_above)
  poly_upper_bound_pts = get_poly_pts(poly_upper_bound_c2, poly_upper_bound_c1, poly_upper_bound_c0, min_x, confirm_to_x)
  poly_upper_bound_str = get_poly_str(poly_upper_bound_c2, poly_upper_bound_c1, poly_upper_bound_c0)
  poly_upper_bound = lambda x: poly_upper_bound_c2 * x**2 + poly_upper_bound_c1 * x + poly_upper_bound_c0
  poly_upper_bound_confirmed = bounds_above(confirm_pts, poly_upper_bound)
poly_lft_confirm_r2 = r2_score(confirm_y, poly_lft_obj(confirm_x))


# right endpoint lower bound
eps = -0.001

lower_bound_list = []

# first method
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_0(rgh, min_x, max_x, eps, bounds_below)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


# second method
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_1(rgh, min_x, max_x, eps, bounds_below)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


# third method
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_2(rgh, min_x, max_x, eps, bounds_below)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


# fourth method
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_3(rgh, min_x, max_x, eps, bounds_below)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


# fifth method
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_4(rgh, min_x, max_x, eps, bounds_below)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


# sixth method
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_5(rgh, min_x, max_x, eps, bounds_below)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


# seventh method
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_6(rgh, min_x, max_x, eps, bounds_below)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


# fourth method - dependent convergence
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_3(rgh, min_x, max_x, eps, bounds_below, dependent_convergence=True)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


# fifth method - dependent convergence
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_4(rgh, min_x, max_x, eps, bounds_below, dependent_convergence=True)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


# sixth method - dependent convergence
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_5(rgh, min_x, max_x, eps, bounds_below, dependent_convergence=True)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


# seventh method - dependent convergence
poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0 = get_best_poly_bound_6(rgh, min_x, max_x, eps, bounds_below, dependent_convergence=True)
poly_lower_bound_str = get_poly_str(poly_lower_bound_c2, poly_lower_bound_c1, poly_lower_bound_c0)
poly_lower_bound = lambda x: poly_lower_bound_c2 * x**2 + poly_lower_bound_c1 * x + poly_lower_bound_c0
if bounds_below(confirm_pts, poly_lower_bound):
  poly_rgh_confirm_r2 = r2_score(confirm_y, [poly_lower_bound(k) for k in confirm_x])
  lower_bound_list.append((poly_rgh_confirm_r2, poly_lower_bound, poly_lower_bound_str))


if len(lower_bound_list) > 0:
  print(sorted(lower_bound_list, key=lambda x: x[0]))
  f = sorted(lower_bound_list, key=lambda x: x[0])[-1][1]
  poly_lower_bound_pts_y = [f(k) for k in confirm_x]
  poly_lower_bound_str = sorted(lower_bound_list, key=lambda x: x[0])[-1][2]
  poly_lower_bound_confirmed = True
else:
  poly_lower_bound_confirmed = False


poly_rgh_confirm_r2 = r2_score(confirm_y, poly_rgh_obj(confirm_x))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([k[0] for k in rmx], [k[1] for k in rmx], '*', label='running max of critical cache size for b = 3')

if poly_upper_bound_confirmed:
  ax.plot([k[0] for k in poly_upper_bound_pts], [k[1] for k in poly_upper_bound_pts], label='upper bound: {}'.format(poly_upper_bound_str))

if poly_lower_bound_confirmed:
  ax.plot(confirm_x, poly_lower_bound_pts_y, label='lower bound: {}'.format(poly_lower_bound_str))

plt.xlabel('problem size (n)')
plt.ylabel('critical cache size')
t = 'LCS v2 traceback, LRU, critical cache size (csize)'
t+= '\nInstance type: a = ceiling(n/3)'
t+= '\n1st sequence: 0^a 1^a 2^(n-2a)'
t+= '\n2nd sequence: 2^(n-2a) 1^a 0^a'
ax.legend(loc='upper left')
plt.title(t)
plt.show()
