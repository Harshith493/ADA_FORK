import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import numpy as np
from fork.phase6 import clip_interval

np.random.seed(0)
# generate 1000 random intervals
lower = np.random.uniform(-20, 60, size=1000)
upper = lower + np.random.uniform(0, 60, size=1000)
raw_width = upper - lower

# clip to [0, 70]
lo_clip, hi_clip = clip_interval(lower, upper, 0.0, 70.0)
clipped_width = hi_clip - lo_clip

print('raw mean width:', raw_width.mean())
print('clipped mean width:', clipped_width.mean())
print('all clipped widths <= raw widths:', np.all(clipped_width <= raw_width))
print('all clipped lower >= 0 and upper <= 70:', np.all(lo_clip >= 0) and np.all(hi_clip <= 70))


np.random.seed(0)
# generate 1000 random intervals
lower = np.random.uniform(-20, 60, size=1000)
upper = lower + np.random.uniform(0, 60, size=1000)
raw_width = upper - lower

# clip to [0, 70]
lo_clip, hi_clip = clip_interval(lower, upper, 0.0, 70.0)
clipped_width = hi_clip - lo_clip

print('raw mean width:', raw_width.mean())
print('clipped mean width:', clipped_width.mean())
print('all clipped widths <= raw widths:', np.all(clipped_width <= raw_width))
print('all clipped lower >= 0 and upper <= 70:', np.all(lo_clip >= 0) and np.all(hi_clip <= 70))
